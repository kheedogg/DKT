import os, sys, time, datetime, pytz
import tensorflow as tf
from sklearn.metrics import roc_curve, auc, accuracy_score
from load_data import OriginalInputProcessor
from model import Model
import numpy as np
import pandas as pd
from tqdm import tqdm

SPLIT_MSG = "***********"


def _seq_length(sequence):
    """
    This function return the sequence length of each x in the batch.
    :param sequence: the batch sequence of shape [batch_size, num_steps, feature_size]
    :return length: A tensor of shape [batch_size]
    """
    used = np.sign(np.max(np.abs(sequence), 2))
    seq_length = np.sum(used, 1)
    # seq_length = np.cast(seq_length, tf.int32)
    return seq_length


class DKT(object):
    def __init__(self, sess, data_train, data_valid, data_test, num_problems, network_config, 
                 save_dir_prefix='./', num_runs=5, num_epochs=500, keep_prob=0.5, logging=True, 
                 save=True):
        self.sess = sess

        self.data_train = data_train
        self.data_valid = data_valid
        self.data_test = data_test
        self.num_problems = num_problems
        self.network_config = network_config
        self.model = Model(num_problems=num_problems, **network_config)
        self.keep_prob = keep_prob
        self.num_epochs = num_epochs
        self.num_runs = num_runs
        self.run_count = 0

        cell_type_str = repr(network_config['rnn_cell']).split('.')[-1][:-6]
        layer_structure_str = "-".join([str(i) for i in network_config['hidden_layer_structure']])
        model_name = self.model_name = cell_type_str + '-' + layer_structure_str
        save_dir_name = ""

        save_dir_name = 'n{}.lo{}.lw1{}.lw2{}'.format(layer_structure_str,
                                                    network_config['lambda_o'],
                                                    network_config['lambda_w1'],
                                                    network_config['lambda_w2'])
        save_dir_name += "/"

        self.ckpt_save_dir = os.path.join(save_dir_prefix, 'model')
        self.log_save_dir = save_dir_prefix
        # print('ckpt_save_dir: ', self.ckpt_save_dir)
        # print('log_save_dir: ', self.log_save_dir)

        if not os.path.exists(self.log_save_dir):
            os.makedirs(self.log_save_dir)
        self.log_file_path = os.path.join(self.log_save_dir, "model_train.log")
        self.logging = logging
        self.save = save

        self._log("Network Configuration:")
        for k, v in network_config.items():
            log_msg = "{}: {}".format(k, v)
            self._log(log_msg)
        self._log("Num of problems: {}".format(num_problems))
        self._log("Num of run: {}".format(num_runs))
        self._log("Max num of run: {}".format(num_epochs))
        self._log("Keep Prob: {}".format(keep_prob))

    def train(self):
        data = self.data_train
        model = self.model
        keep_prob = self.keep_prob
        sess = self.sess

        loss = 0.0
        y_pred = []
        y_true = []
        iteration = 1
        for batch_idx in tqdm(range(data.num_batches), desc="train batch..."):
            X_batch, y_seq_batch, y_corr_batch = data.next_batch()
            feed_dict = {
                model.X: X_batch,
                model.y_seq: y_seq_batch,
                model.y_corr: y_corr_batch,
                model.keep_prob: keep_prob,
            }
            _, _target_preds, _target_labels, _loss = sess.run(
                [model.train_op, model.target_preds, model.target_labels, model.loss],
                feed_dict=feed_dict
            )
            y_pred += [p for p in _target_preds]
            y_true += [t for t in _target_labels]
            loss = (iteration - 1) / iteration * loss + _loss / iteration
            iteration += 1
        try:
            acc_score = accuracy_score(np.array(y_true), np.round(y_pred))
            fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
            auc_score = auc(fpr, tpr)
        except ValueError:
            self._log("Value Error is encountered during finding the acc_score and auc_score. Assign the AUC to 0 now.")
            acc_score = 0.0
            auc_score = 0.0
            loss = 999999.9

        return acc_score, auc_score, loss

    def evaluate(self, mode='valid', makefile=False):
        if mode == 'train':
            data = self.data_train
        elif mode == 'valid':
            data = self.data_valid
        else:
            data = self.data_test

        data.reset_cursor()
        model = self.model
        sess = self.sess
        
        total_confusion_matrix = list()
        y_pred = []
        y_true = []
        y_pred_current = []
        y_true_current = []
        iteration = 1
        loss = 0.0
        acc_score = 0.0
        auc_score_current = 0.0
        auc_score = 0.0
        for batch_idx in tqdm(range(data.num_batches), desc=f"{mode} batch..."):
            X_batch, y_seq_batch, y_corr_batch = data.next_batch()
            feed_dict = {
                model.X: X_batch,
                model.y_seq: y_seq_batch,
                model.y_corr: y_corr_batch,
                model.keep_prob: 1,
            }
            _target_preds, _target_labels, _target_preds_current, _target_labels_current, _loss = sess.run(
                [model.target_preds,
                 model.target_labels,
                 model.target_preds_current,
                 model.target_labels_current,
                 model.loss],
                feed_dict=feed_dict
            )
            
            y_pred += [_target_preds[p] for p in range(len(_target_preds))]
            y_true += [_target_labels[t] for t in range(len(_target_labels))]
            y_pred_current += [_target_preds_current[p]  for p in range(len(_target_preds_current))]
            y_true_current += [_target_labels_current[t]  for t in range(len(_target_labels_current))]
            loss = (iteration - 1) / iteration * loss + _loss / iteration
            iteration += 1

            confusion = list()
            
            if makefile:
                for i in range(len(_target_preds)):
                     _target_preds[i] = int(np.round(_target_preds[i]))
                     _target_labels[i] = int(_target_labels[i])
                     if _target_preds[i] == _target_labels[i]:
                         if _target_preds[i] == 0: # (true, pred) = (0,0) -> TN
                            confusion.append('TN')
                         else: #(true, pred) = (1, 1)
                            confusion.append('TP')
                     else:
                         if _target_preds[i] == 0: #(true, pred) = (1, 0) ->FN
                             confusion.append('FN')
                         else:
                             confusion.append('FP')
    
                total_confusion_matrix.append(confusion)
            
        try:
            acc_score = accuracy_score(np.array(y_true), np.round(y_pred))
            fpr, tpr, thres = roc_curve(y_true, y_pred, pos_label=1)
            auc_score = auc(fpr, tpr)
            fpr, tpr, thres = roc_curve(y_true_current, y_pred_current, pos_label=1)
            auc_score_current = auc(fpr, tpr)
        except ValueError:
            self._log("Value Error is encountered during finding the auc_score. Assign the AUC to 0 now.")
            acc_score = 0.0
            auc_score = 0.0
            auc_score_current = 0.0
            loss = 999999.9
            
	#confusion matrix of students
        if makefile:
            filename= './results/confusion_information_'+ str(mode)+ '.csv'
            tp, tn, fp, fn = 0, 0, 0, 0
            f = open(filename, 'w')
            for i in range(len(total_confusion_matrix)):
                data = ' ,' + ','.join(total_confusion_matrix[i]) + '\n' 
                tp += total_confusion_matrix[i].count('TP')
                tn += total_confusion_matrix[i].count('TN')
                fp += total_confusion_matrix[i].count('FP')
                fn += total_confusion_matrix[i].count('FN')
                f.write(data)
            f.close()
        
        print('ACC :{}, AUC:{}, AUC_SCORE_CURRENT:{}'.format(acc_score, auc_score, auc_score_current))

        return acc_score, auc_score, auc_score_current, loss, total_confusion_matrix

    def run_optimization(self):
        num_epochs = self.num_epochs
        num_runs = self.num_runs
        sess = self.sess

        total_auc = 0.0
        self.accs = []
        self.aucs = []
        self.test_accs = []
        self.test_aucs = []
        self.test_aucs_current = []
        self.aucs_current = []
        self.wavinesses_l1 = []
        self.wavinesses_l2 = []
        self.consistency_m1 = []
        self.consistency_m2 = []
        for run_idx in range(num_runs):
            print("\n{0} th repeat training ..".format(run_idx+1))
            self.run_count = run_idx
            # sess.run(tf.global_variables_initializer())
            sess.run(tf.compat.v1.global_variables_initializer())
            acc_test = 0.0
            auc_test = 0.0
            auc_current_test = 0.0
            best_valid_acc = 0.0
            best_valid_auc = 0.0
            best_valid_auc_current = 0.0 # the auc_current when the test_auc is the best.
            best_waviness_l1 = 0.0
            best_waviness_l2 = 0.0
            best_consistency_m1 = 0.0
            best_consistency_m2 = 0.0

            best_epoch_idx = 0
            for epoch_idx in range(num_epochs):
                epoch_start_time = time.time()
                local_tz = pytz.timezone('Asia/Seoul')
                date = datetime.datetime.now(datetime.timezone.utc)
                current_time = date.astimezone(local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
                acc_train, auc_train, loss_train = self.train()
                self._log('Current time : {}'.format(current_time))

                self._log(
                    'Epoch {0:>4}, Train ACC: {1:.5}, Train AUC: {2:.5}, Train Loss: {3:.5}'.format(epoch_idx + 1, acc_train, auc_train, loss_train))

                acc_valid, auc_valid, auc_current_valid, loss_valid, _ = self.evaluate('valid')
                valid_msg = "Epoch {:>4}, Valid ACC: {:.5}, Valid AUC: {:.5}, Valid AUC Curr: {:.5}, Valid Loss: {:.5}".format(
                    epoch_idx + 1,
                    acc_valid,
                    auc_valid,
                    auc_current_valid,
                    loss_valid)

                if auc_train == 0 and auc_valid == 0:
                    self._log("ValueError occur, break the epoch loop.")
                    break

                if acc_valid > best_valid_acc:
                    valid_msg += "*"
                    best_epoch_idx = epoch_idx
                    best_valid_acc = acc_valid
                    best_valid_auc = auc_valid
                    best_valid_auc_current = auc_current_valid
                    best_waviness_l1, best_waviness_l2 = self.waviness('valid')

                    acc_test, auc_test, auc_current_test, loss_test, _ = self.evaluate('test')
                    valid_msg += "\nEpoch {:>4}, Test ACC: {:.5}, Test AUC: {:.5}, Test AUC Curr: {:.5}, Test Loss: {:.5}".format(
                        epoch_idx + 1,
                        acc_test,
                        auc_test,
                        auc_current_test,
                        loss_test)

                    m1, m2 = self.consistency('valid')
                    best_consistency_m1 = m1
                    best_consistency_m2 = m2

                    valid_msg += "\nw_l1: {0:5}, w_l2: {1:5}".format(best_waviness_l1, best_waviness_l2)
                    valid_msg += "\nm1: {0:5}, m2: {1:5}".format(best_consistency_m1, best_consistency_m2)
                    if self.save:
                        valid_msg += ". Saving the model"
                        self.save_model()
                    
                self._log(valid_msg)

                epoch_end_time = time.time()
                self._log("time used for this epoch: {0}s".format(epoch_end_time - epoch_start_time))
                self._log(SPLIT_MSG)

                if epoch_idx - best_epoch_idx >= 10:
                    self._log("No improvement shown in 10 epochs. Quit Training.")
                    break
                sys.stdout.flush()
                self.data_train.shuffle()
                
            self._log("The best validation result occured at: {0}-th epoch, with validation ACC: {1:.5} and AUC: {2:.5}".format(
                best_epoch_idx + 1, best_valid_acc, best_valid_auc))
            self._log("The best testing result occured at: {0}-th epoch, with testing ACC: {1:.5} and AUC: {2:.5}".format(
                best_epoch_idx + 1, acc_test, auc_test))
            
            self._log(SPLIT_MSG * 3)
            self.wavinesses_l1.append(best_waviness_l1)
            self.wavinesses_l2.append(best_waviness_l2)
            self.accs.append(best_valid_acc)
            self.aucs.append(best_valid_auc)
            self.test_accs.append(acc_test)
            self.test_aucs.append(auc_test)
            self.aucs_current.append(best_valid_auc_current)
            self.test_aucs_current.append(auc_current_test)
            self.consistency_m1.append(best_consistency_m1)
            self.consistency_m2.append(best_consistency_m2)
        avg_acc = np.average(self.accs)
        avg_auc = np.average(self.aucs)
        avg_test_acc = np.average(self.test_accs)
        avg_test_auc = np.average(self.test_aucs)
        avg_auc_current = np.average(self.aucs_current)
        avg_test_auc_current = np.average(self.test_aucs_current)
        avg_waviness_l1 = np.average(self.wavinesses_l1)
        avg_waviness_l2 = np.average(self.wavinesses_l2)
        avg_consistency_m1 = np.average(self.consistency_m1)
        avg_consistency_m2 = np.average(self.consistency_m2)

        self._log("average validation ACC for {0} runs: {1}".format(num_runs, avg_acc))
        self._log("average validation AUC for {0} runs: {1}".format(num_runs, avg_auc))
        self._log("average validation AUC Current for {0} runs: {1}".format(num_runs, avg_auc_current))
        self._log("\naverage waviness-l1 for {0} runs: {1}".format(num_runs, avg_waviness_l1))
        self._log("average waviness-l2 for {0} runs: {1}".format(num_runs, avg_waviness_l2))
        self._log("average consistency_m1 for {0} runs: {1}".format(num_runs, avg_consistency_m1))
        self._log("average consistency_m1 for {0} runs: {1}".format(num_runs, avg_consistency_m2))
        
        self._log("\ntest ACC for {0} runs : {1}".format(num_runs, self.test_accs))
        self._log("test AUC for {0} runs : {1}".format(num_runs, self.test_aucs))
        self._log("\naverage test ACC for {0} runs: {1}".format(num_runs, avg_test_acc))
        self._log("average test AUC for {0} runs: {1}".format(num_runs, avg_test_auc))
        self._log("average test AUC Current for {0} runs: {1}\n".format(num_runs, avg_test_auc_current))
        
        self._log("latex: \n" + self.auc_summary_in_latex())
        return avg_test_acc

    def save_model(self):
        save_dir = self.ckpt_save_dir
        sess = self.sess
        # saver = tf.train.Saver()
        saver = tf.compat.v1.train.Saver()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, self.model_name)
        pb_save_path = os.path.join(save_dir, 'model.pb')
        pb_txt_save_path = os.path.join(save_dir, 'model.pbtxt')
        saver.save(sess=sess, save_path=save_path)
        tf.io.write_graph(sess.graph_def, '.', pb_save_path, as_text=False)
        tf.io.write_graph(sess.graph_def, '.', pb_txt_save_path, as_text=True)

    def load_model(self):
        save_dir = os.path.join(self.ckpt_save_dir, 'run_{}'.format(self.run_count), self.model_name)
        sess = self.sess
        # saver = tf.train.Saver()
        saver = tf.compat.v1.train.Saver()
        save_path = os.path.join(save_dir, self.model_name)
        if os.path.exists(save_path):
            saver.restore(sess=sess, save_path=save_path)
        else:
            self._log("No model found at {}".format(save_path))

    def get_hidden_layer_output(self, problem_seqs, correct_seqs, layer):
        model = self.model
        sess = self.sess
        num_layer = len(model.hidden_layer_structure)
        assert layer < num_layer, "There are only {0} layers. indexed from 0.".format(num_layer)

        input_processor = OriginalInputProcessor()
        X, y_seq, y_corr = input_processor.process_problems_and_corrects(problem_seqs=problem_seqs,
                                                                         correct_seqs=correct_seqs,
                                                                         num_problems=self.num_problems)

        feed_dict = {
            model.X: X,
            model.y_seq: y_seq,
            model.y_corr: y_corr,
            model.keep_prob: 1.0,
        }

        hidden_layers_outputs = sess.run(
            model.hidden_layers_outputs,
            feed_dict=feed_dict
        )

        result = hidden_layers_outputs[layer]
        return result

    def get_output_layer(self, problem_seqs, correct_seqs):
        model = self.model
        sess = self.sess

        input_processor = OriginalInputProcessor()
        X, y_seq, y_corr = input_processor.process_problems_and_corrects(problem_seqs=problem_seqs,
                                                                         correct_seqs=correct_seqs,
                                                                         num_problems=self.num_problems,
                                                                         is_train=False)

        feed_dict = {
            model.X: X,
            model.y_seq: y_seq,
            model.y_corr: y_corr,
            model.keep_prob: 1.0,
        }

        pred_seqs = sess.run(
            model.preds,
            feed_dict=feed_dict
        )

        return pred_seqs

    def _log(self, log_msg):
        print(log_msg)
        if self.logging:
            with open(self.log_file_path, "a+") as f:
                f.write(log_msg + '\n')

    def auc_summary_in_latex(self):
        # def mean_confidence_interval(data, confidence=0.95):
        #     import scipy.stats as st
        #     import numpy as np
        #     a = 1.0 * np.array(data)
        #     n = len(a)
        #     m, se = np.mean(a), st.sem(a)
        #     h = se * st.t.ppf((1 + confidence) / 2., n - 1)
        #     return m, h
        #
        # assert len(aucs) > 1, "There should be at least two auc scores to find the interval."
        cell_type_str = repr(self.network_config['rnn_cell']).split('.')[-1][:-6]
        num_layers_str = str(len(self.network_config['hidden_layer_structure']))
        layer_structure_str = ", ".join([str(i) for i in self.network_config['hidden_layer_structure']])

        # experiment result
        acc_mean = np.average(self.accs)
        acc_std = np.std(self.accs)
        
        auc_mean = np.average(self.aucs)
        auc_std = np.std(self.aucs)

        auc_current_mean = np.average(self.aucs_current)
        auc_current_std = np.std(self.aucs_current)

        waviness_l1_mean = np.average(self.wavinesses_l1)
        waviness_l1_std = np.std(self.wavinesses_l1)

        waviness_l2_mean = np.average(self.wavinesses_l2)
        waviness_l2_std = np.std(self.wavinesses_l2)

        consistency_m1_mean = np.average(self.consistency_m1)
        consistency_m1_std = np.std(self.consistency_m1)

        consistency_m2_mean = np.average(self.consistency_m2)
        consistency_m2_std = np.std(self.consistency_m2)

        # cell_type & num. layer & layer_structure & learning rate & keep prob & Avg. AUC & Avg. Waviness
        # LSTM & 1 & (200,) & 0.0100 & 0.500 & 0.010 & 0.82500 $\pm$ 0.000496\\
        result_cols = [
            'cell_type',
            'num. layer',
            'layer_structure',
            'learning rate',
            'keep prob.',
            '$\lambda_o$',
            '$\lambda_{w_1}$',
            '$\lambda_{w_2}$',
            'Avg. ACC(N)',
            'Avg. AUC(N)',
            'Avg. AUC(C)',
            'Avg. $w_1$',
            'Avg. $w_2$',
            'Avg. $m_1$',
            'Avg. $m_2$',
        ]

        result_data = [
            cell_type_str,
            num_layers_str,
            layer_structure_str,
            "{:.4f}".format(self.network_config['learning_rate']),
            "{:.4f}".format(self.network_config['keep_prob']),
            "{:.4f}".format(self.network_config['lambda_o']),
            "{:.4f}".format(self.network_config['lambda_w1']),
            "{:.4f}".format(self.network_config['lambda_w2']),
            "{} $\pm$ {}".format(acc_mean, acc_std),
            "{} $\pm$ {}".format(auc_mean, auc_std),
            "{} $\pm$ {}".format(auc_current_mean, auc_current_std),
            "{} $\pm$ {}".format(waviness_l1_mean, waviness_l1_std),
            "{} $\pm$ {}".format(waviness_l2_mean, waviness_l2_std),
            "{} $\pm$ {}".format(consistency_m1_mean, consistency_m1_std),
            "{} $\pm$ {}".format(consistency_m2_mean, consistency_m2_std),
        ]

        latex_str = " & ".join(result_cols)
        latex_str += "\\\\ \n"

        latex_str += " & ".join(result_data)
        latex_str += "\\\\ \n"
        return latex_str

    def plot_output_layer(self, problem_seq, correct_seq, target_problem_ids=None):
        import matplotlib.pyplot as plt
        import seaborn as sns
        problem_ids_answered = sorted(set(problem_seq))
        if target_problem_ids is None:
            target_problem_ids = problem_ids_answered

        # get_output_layer return output in shape (1, 38, 124)
        output = self.get_output_layer(problem_seqs=[problem_seq], correct_seqs=[correct_seq])[0]  # shape (38, 124)
        output = output[:, target_problem_ids]  # shape (38, ?)
        output = np.transpose(output)  # shape (?, 38)

        y_labels = target_problem_ids
        x_labels = ["({},{})".format(p, c) for p, c in zip(problem_seq, correct_seq)]
        df = pd.DataFrame(output)
        df.columns = x_labels
        df.index = y_labels

        return sns.heatmap(df, vmin=0, vmax=1, cmap=plt.cm.Blues)

    def plot_hidden_layer(self, problem_seq, correct_seq, layer):
        import matplotlib.pyplot as plt
        import seaborn as sns
        output = self.get_hidden_layer_output(problem_seqs=[problem_seq], correct_seqs=[correct_seq], layer=layer)
        output = output[0]  # ignore the batch_idx
        output = np.transpose(output)

        y_labels = range(output.shape[0])
        x_labels = ["({},{})".format(p, c) for p, c in zip(problem_seq, correct_seq)]
        df = pd.DataFrame(output)
        df.columns = x_labels
        df.index = y_labels

        return sns.heatmap(df, cmap='RdBu')

    def waviness(self, mode='valid'):
        if mode == 'train':
            data = self.data_train
            is_train=True
        elif mode == 'valid':
            data = self.data_valid
            is_train=False
        else:
            data = self.data_test
            is_train=False
        data.reset_cursor()
        model = self.model
        sess = self.sess

        waviness_l1 = 0.0
        waviness_l2 = 0.0
        total_num_steps = 0.0
        for batch_idx in range(data.num_batches):
            # print('batch:', batch_idx, end='\r')
            X_batch, y_seq_batch, y_corr_batch = data.next_batch(is_train)
            feed_dict = {
                model.X: X_batch,
                model.y_seq: y_seq_batch,
                model.y_corr: y_corr_batch,
                model.keep_prob: 1,
            }
            _waviness_l1, _waviness_l2, _total_num_steps = sess.run(
                [model.waviness_l1,
                 model.waviness_l2,
                 model.total_num_steps],
                feed_dict=feed_dict
            )
            waviness_l1 += _waviness_l1 * _total_num_steps
            waviness_l2 += _waviness_l2 * _total_num_steps
            total_num_steps += _total_num_steps
        waviness_l1 /= total_num_steps
        waviness_l2 /= total_num_steps
        waviness_l2 = np.sqrt(waviness_l2)

        return waviness_l1, waviness_l2


    def waviness_np(self, mode='valid'):
        if mode == 'train':
            data = self.data_train
            is_train=True
        elif mode == 'valid':
            data = self.data_valid
            is_train=False
        else:
            data = self.data_test
            is_train=False
        data.reset_cursor()
        model = self.model
        sess = self.sess

        waviness_l1 = 0.0
        waviness_l2 = 0.0
        total_num_steps = 0.0
        for batch_idx in range(data.num_batches):
            X_batch, y_seq_batch, y_corr_batch = data.next_batch(is_train)

            feed_dict = {
                model.X: X_batch,
                model.y_seq: y_seq_batch,
                model.y_corr: y_corr_batch,
                model.keep_prob: 1,
            }
            pred_seqs = sess.run(
                model.preds,
                feed_dict=feed_dict
            )

            # finding w1, w2 for this batch
            w1 = np.sum(np.abs(pred_seqs[:, 1:, :] - pred_seqs[:, :-1, :]))
            w2 = np.sum(np.square(pred_seqs[:, 1:, :] - pred_seqs[:, :-1, :]))

            seq_length_batch = np.sum(_seq_length(y_seq_batch[:, 1:, :]))
            waviness_l1 += w1
            waviness_l2 += w2
            total_num_steps += seq_length_batch

            # print('batch:{}, w1:{}, w2:{}, length:{}'.format(batch_idx, w1, w2, seq_length_batch), end='\r')

        waviness_l1 /= (total_num_steps * data.num_problems)
        waviness_l2 /= (total_num_steps * data.num_problems)
        waviness_l2 = np.sqrt(waviness_l2)

        return waviness_l1, waviness_l2

    def _reconstruction_accurarcy(self, mode='valid'):
        if mode == 'train':
            data = self.data_train
        elif mode == 'valid':
            data = self.data_valid
        else:
            data = self.data_test
        data.reset_cursor()

        problem_seqs = data.problem_seqs
        correct_seqs = data.correct_seqs
        num_interactions = 0
        sign_diff_score = 0
        diff_score = 0
        for i in range(len(problem_seqs)):
            if i%20 == 0:
                print(i, end='\r')
            problem_seq = problem_seqs[i]
            correct_seq = correct_seqs[i]
            outputs = self.get_output_layer([problem_seq], [correct_seq]) # shape: (batch, time, num_problems)

            for j in range(1, len(problem_seq)): # exclude the prediction of the first output
                target_id = problem_seq[j]
                label = correct_seq[j]
                score = 1.0 if label==1 else -1.0

                prev_pred = outputs[0][j-1][target_id]
                curr_pred = outputs[0][j][target_id]
                pred_diff = curr_pred - prev_pred
                pred_sign_diff = np.sign(pred_diff)

                sign_diff_score += pred_sign_diff * score
                diff_score += pred_diff * score
                num_interactions += 1
        return (sign_diff_score, diff_score, num_interactions)

    def consistency(self, mode='valid'):
        if mode == 'train':
            data = self.data_train
            is_train=True
        elif mode == 'valid':
            data = self.data_valid
            is_train=False
        else:
            data = self.data_test
            is_train=False
        data.reset_cursor()
        model = self.model
        sess = self.sess

        consistency_m1 = 0.0
        consistency_m2 = 0.0
        total_num_steps = 0.0
        for batch_idx in range(data.num_batches):
            # X_batch: one hot encoded (q_t, a_t)
            # y_seq_batch: one hot encoded (q_t), \deltadm{q_t}
            # y_corr_batch: one hot encoded (a_t)
            X_batch, y_seq_batch, y_corr_batch = data.next_batch(is_train)
            seq_length_batch = np.sum(_seq_length(y_seq_batch[:, 1:, :]))

            feed_dict = {
                model.X: X_batch,
                model.y_seq: y_seq_batch,
                model.y_corr: y_corr_batch,
                model.keep_prob: 1,
            }
            pred_seqs = sess.run(
                model.preds,
                feed_dict=feed_dict
            )

            # finding m1, m2 for this batch
            base = y_seq_batch[:, 1:, :].copy()
            base[:] = -1.0
            coefficient = np.sum( (np.power(base, 1 - y_corr_batch[:, 1:, :])) * y_seq_batch[:, 1:, :], axis=2)

            m1 = np.sum(
                coefficient * np.sign(np.sum(
                    (pred_seqs[:, 1:, :] - pred_seqs[:, :-1, :]) * y_seq_batch[:, 1:, :], #y_t-y_{t-1} \dot
                    axis=2
                ))
            )
            m2 = np.sum(
                coefficient * np.sum(
                    (pred_seqs[:, 1:, :] - pred_seqs[:, :-1, :]) * y_seq_batch[:, 1:, :],
                    axis=2
                )
            )

            consistency_m1 += m1
            consistency_m2 += m2
            total_num_steps += seq_length_batch

        consistency_m1 /= (total_num_steps)
        consistency_m2 /= (total_num_steps)

        return consistency_m1, consistency_m2
