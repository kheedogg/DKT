
I
XPlaceholder*
dtype0**
shape!:�������������������
M
y_seqPlaceholder*
dtype0**
shape!:�������������������
N
y_corrPlaceholder*
dtype0**
shape!:�������������������
<
keep_prob/inputConst*
dtype0*
valueB
 *  �?
N
	keep_probPlaceholderWithDefaultkeep_prob/input*
dtype0*
shape: 

AbsAbsX*
T0
?
Max/reduction_indicesConst*
dtype0*
value	B :
L
MaxMaxAbsMax/reduction_indices*
T0*

Tidx0*
	keep_dims( 

SignSignMax*
T0
?
Sum/reduction_indicesConst*
dtype0*
value	B :
M
SumSumSignSum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
9
CastCastSum*

DstT0*

SrcT0*
Truncate( 
T
'hidden_layer_0/DropoutWrapperInit/ConstConst*
dtype0*
valueB
 *  �?
V
)hidden_layer_0/DropoutWrapperInit/Const_1Const*
dtype0*
valueB
 *  �?
A
hidden_layer_0/rnn/RankConst*
dtype0*
value	B :
H
hidden_layer_0/rnn/range/startConst*
dtype0*
value	B :
H
hidden_layer_0/rnn/range/deltaConst*
dtype0*
value	B :
�
hidden_layer_0/rnn/rangeRangehidden_layer_0/rnn/range/starthidden_layer_0/rnn/Rankhidden_layer_0/rnn/range/delta*

Tidx0
W
"hidden_layer_0/rnn/concat/values_0Const*
dtype0*
valueB"       
H
hidden_layer_0/rnn/concat/axisConst*
dtype0*
value	B : 
�
hidden_layer_0/rnn/concatConcatV2"hidden_layer_0/rnn/concat/values_0hidden_layer_0/rnn/rangehidden_layer_0/rnn/concat/axis*
N*
T0*

Tidx0
]
hidden_layer_0/rnn/transpose	TransposeXhidden_layer_0/rnn/concat*
T0*
Tperm0
=
"hidden_layer_0/rnn/sequence_lengthIdentityCast*
T0
X
hidden_layer_0/rnn/ShapeShapehidden_layer_0/rnn/transpose*
T0*
out_type0
T
&hidden_layer_0/rnn/strided_slice/stackConst*
dtype0*
valueB:
V
(hidden_layer_0/rnn/strided_slice/stack_1Const*
dtype0*
valueB:
V
(hidden_layer_0/rnn/strided_slice/stack_2Const*
dtype0*
valueB:
�
 hidden_layer_0/rnn/strided_sliceStridedSlicehidden_layer_0/rnn/Shape&hidden_layer_0/rnn/strided_slice/stack(hidden_layer_0/rnn/strided_slice/stack_1(hidden_layer_0/rnn/strided_slice/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
u
Khidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dimConst*
dtype0*
value	B : 
�
Ghidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims
ExpandDims hidden_layer_0/rnn/strided_sliceKhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dim*
T0*

Tdim0
p
Bhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ConstConst*
dtype0*
valueB:f
r
Hhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat/axisConst*
dtype0*
value	B : 
�
Chidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concatConcatV2Ghidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDimsBhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ConstHhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat/axis*
N*
T0*

Tidx0
u
Hhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
dtype0*
valueB
 *    
�
Bhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zerosFillChidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concatHhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros/Const*
T0*

index_type0
w
Mhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dimConst*
dtype0*
value	B : 
�
Ihidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1
ExpandDims hidden_layer_0/rnn/strided_sliceMhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dim*
T0*

Tdim0
r
Dhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Const*
dtype0*
valueB:f
w
Mhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dimConst*
dtype0*
value	B : 
�
Ihidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2
ExpandDims hidden_layer_0/rnn/strided_sliceMhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dim*
T0*

Tdim0
r
Dhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Const*
dtype0*
valueB:f
t
Jhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
dtype0*
value	B : 
�
Ehidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2Ihidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2Dhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Jhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axis*
N*
T0*

Tidx0
w
Jhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
dtype0*
valueB
 *    
�
Dhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1FillEhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat_1Jhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0
w
Mhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dimConst*
dtype0*
value	B : 
�
Ihidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3
ExpandDims hidden_layer_0/rnn/strided_sliceMhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dim*
T0*

Tdim0
r
Dhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/Const_3Const*
dtype0*
valueB:f
`
hidden_layer_0/rnn/Shape_1Shape"hidden_layer_0/rnn/sequence_length*
T0*
out_type0
`
hidden_layer_0/rnn/stackPack hidden_layer_0/rnn/strided_slice*
N*
T0*

axis 
�
hidden_layer_0/rnn/EqualEqualhidden_layer_0/rnn/Shape_1hidden_layer_0/rnn/stack*
T0*
incompatible_shape_error(
F
hidden_layer_0/rnn/ConstConst*
dtype0*
valueB: 
n
hidden_layer_0/rnn/AllAllhidden_layer_0/rnn/Equalhidden_layer_0/rnn/Const*

Tidx0*
	keep_dims( 
�
hidden_layer_0/rnn/Assert/ConstConst*
dtype0*S
valueJBH BBExpected shape for Tensor hidden_layer_0/rnn/sequence_length:0 is 
Z
!hidden_layer_0/rnn/Assert/Const_1Const*
dtype0*!
valueB B but saw shape: 
�
'hidden_layer_0/rnn/Assert/Assert/data_0Const*
dtype0*S
valueJBH BBExpected shape for Tensor hidden_layer_0/rnn/sequence_length:0 is 
`
'hidden_layer_0/rnn/Assert/Assert/data_2Const*
dtype0*!
valueB B but saw shape: 
�
 hidden_layer_0/rnn/Assert/AssertAsserthidden_layer_0/rnn/All'hidden_layer_0/rnn/Assert/Assert/data_0hidden_layer_0/rnn/stack'hidden_layer_0/rnn/Assert/Assert/data_2hidden_layer_0/rnn/Shape_1*
T
2*
	summarize
z
hidden_layer_0/rnn/CheckSeqLenIdentity"hidden_layer_0/rnn/sequence_length!^hidden_layer_0/rnn/Assert/Assert*
T0
Z
hidden_layer_0/rnn/Shape_2Shapehidden_layer_0/rnn/transpose*
T0*
out_type0
V
(hidden_layer_0/rnn/strided_slice_1/stackConst*
dtype0*
valueB: 
X
*hidden_layer_0/rnn/strided_slice_1/stack_1Const*
dtype0*
valueB:
X
*hidden_layer_0/rnn/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
"hidden_layer_0/rnn/strided_slice_1StridedSlicehidden_layer_0/rnn/Shape_2(hidden_layer_0/rnn/strided_slice_1/stack*hidden_layer_0/rnn/strided_slice_1/stack_1*hidden_layer_0/rnn/strided_slice_1/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
Z
hidden_layer_0/rnn/Shape_3Shapehidden_layer_0/rnn/transpose*
T0*
out_type0
V
(hidden_layer_0/rnn/strided_slice_2/stackConst*
dtype0*
valueB:
X
*hidden_layer_0/rnn/strided_slice_2/stack_1Const*
dtype0*
valueB:
X
*hidden_layer_0/rnn/strided_slice_2/stack_2Const*
dtype0*
valueB:
�
"hidden_layer_0/rnn/strided_slice_2StridedSlicehidden_layer_0/rnn/Shape_3(hidden_layer_0/rnn/strided_slice_2/stack*hidden_layer_0/rnn/strided_slice_2/stack_1*hidden_layer_0/rnn/strided_slice_2/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
K
!hidden_layer_0/rnn/ExpandDims/dimConst*
dtype0*
value	B : 
�
hidden_layer_0/rnn/ExpandDims
ExpandDims"hidden_layer_0/rnn/strided_slice_2!hidden_layer_0/rnn/ExpandDims/dim*
T0*

Tdim0
H
hidden_layer_0/rnn/Const_1Const*
dtype0*
valueB:f
J
 hidden_layer_0/rnn/concat_1/axisConst*
dtype0*
value	B : 
�
hidden_layer_0/rnn/concat_1ConcatV2hidden_layer_0/rnn/ExpandDimshidden_layer_0/rnn/Const_1 hidden_layer_0/rnn/concat_1/axis*
N*
T0*

Tidx0
K
hidden_layer_0/rnn/zeros/ConstConst*
dtype0*
valueB
 *    
x
hidden_layer_0/rnn/zerosFillhidden_layer_0/rnn/concat_1hidden_layer_0/rnn/zeros/Const*
T0*

index_type0
H
hidden_layer_0/rnn/Const_2Const*
dtype0*
valueB: 

hidden_layer_0/rnn/MinMinhidden_layer_0/rnn/CheckSeqLenhidden_layer_0/rnn/Const_2*
T0*

Tidx0*
	keep_dims( 
H
hidden_layer_0/rnn/Const_3Const*
dtype0*
valueB: 

hidden_layer_0/rnn/MaxMaxhidden_layer_0/rnn/CheckSeqLenhidden_layer_0/rnn/Const_3*
T0*

Tidx0*
	keep_dims( 
A
hidden_layer_0/rnn/timeConst*
dtype0*
value	B : 
�
hidden_layer_0/rnn/TensorArrayTensorArrayV3"hidden_layer_0/rnn/strided_slice_1*
clear_after_read(*
dtype0*
dynamic_size( *$
element_shape:���������f*
identical_element_shapes(*>
tensor_array_name)'hidden_layer_0/rnn/dynamic_rnn/output_0
�
 hidden_layer_0/rnn/TensorArray_1TensorArrayV3"hidden_layer_0/rnn/strided_slice_1*
clear_after_read(*
dtype0*
dynamic_size( *%
element_shape:����������*
identical_element_shapes(*=
tensor_array_name(&hidden_layer_0/rnn/dynamic_rnn/input_0
k
+hidden_layer_0/rnn/TensorArrayUnstack/ShapeShapehidden_layer_0/rnn/transpose*
T0*
out_type0
g
9hidden_layer_0/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: 
i
;hidden_layer_0/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:
i
;hidden_layer_0/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
valueB:
�
3hidden_layer_0/rnn/TensorArrayUnstack/strided_sliceStridedSlice+hidden_layer_0/rnn/TensorArrayUnstack/Shape9hidden_layer_0/rnn/TensorArrayUnstack/strided_slice/stack;hidden_layer_0/rnn/TensorArrayUnstack/strided_slice/stack_1;hidden_layer_0/rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
[
1hidden_layer_0/rnn/TensorArrayUnstack/range/startConst*
dtype0*
value	B : 
[
1hidden_layer_0/rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
value	B :
�
+hidden_layer_0/rnn/TensorArrayUnstack/rangeRange1hidden_layer_0/rnn/TensorArrayUnstack/range/start3hidden_layer_0/rnn/TensorArrayUnstack/strided_slice1hidden_layer_0/rnn/TensorArrayUnstack/range/delta*

Tidx0
�
Mhidden_layer_0/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3 hidden_layer_0/rnn/TensorArray_1+hidden_layer_0/rnn/TensorArrayUnstack/rangehidden_layer_0/rnn/transpose"hidden_layer_0/rnn/TensorArray_1:1*
T0*/
_class%
#!loc:@hidden_layer_0/rnn/transpose
F
hidden_layer_0/rnn/Maximum/xConst*
dtype0*
value	B :
d
hidden_layer_0/rnn/MaximumMaximumhidden_layer_0/rnn/Maximum/xhidden_layer_0/rnn/Max*
T0
n
hidden_layer_0/rnn/MinimumMinimum"hidden_layer_0/rnn/strided_slice_1hidden_layer_0/rnn/Maximum*
T0
T
*hidden_layer_0/rnn/while/iteration_counterConst*
dtype0*
value	B : 
�
hidden_layer_0/rnn/while/EnterEnter*hidden_layer_0/rnn/while/iteration_counter*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
 hidden_layer_0/rnn/while/Enter_1Enterhidden_layer_0/rnn/time*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
 hidden_layer_0/rnn/while/Enter_2Enter hidden_layer_0/rnn/TensorArray:1*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
 hidden_layer_0/rnn/while/Enter_3EnterBhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
 hidden_layer_0/rnn/while/Enter_4EnterDhidden_layer_0/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
hidden_layer_0/rnn/while/MergeMergehidden_layer_0/rnn/while/Enter&hidden_layer_0/rnn/while/NextIteration*
N*
T0
�
 hidden_layer_0/rnn/while/Merge_1Merge hidden_layer_0/rnn/while/Enter_1(hidden_layer_0/rnn/while/NextIteration_1*
N*
T0
�
 hidden_layer_0/rnn/while/Merge_2Merge hidden_layer_0/rnn/while/Enter_2(hidden_layer_0/rnn/while/NextIteration_2*
N*
T0
�
 hidden_layer_0/rnn/while/Merge_3Merge hidden_layer_0/rnn/while/Enter_3(hidden_layer_0/rnn/while/NextIteration_3*
N*
T0
�
 hidden_layer_0/rnn/while/Merge_4Merge hidden_layer_0/rnn/while/Enter_4(hidden_layer_0/rnn/while/NextIteration_4*
N*
T0
s
hidden_layer_0/rnn/while/LessLesshidden_layer_0/rnn/while/Merge#hidden_layer_0/rnn/while/Less/Enter*
T0
�
#hidden_layer_0/rnn/while/Less/EnterEnter"hidden_layer_0/rnn/strided_slice_1*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
y
hidden_layer_0/rnn/while/Less_1Less hidden_layer_0/rnn/while/Merge_1%hidden_layer_0/rnn/while/Less_1/Enter*
T0
�
%hidden_layer_0/rnn/while/Less_1/EnterEnterhidden_layer_0/rnn/Minimum*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
q
#hidden_layer_0/rnn/while/LogicalAnd
LogicalAndhidden_layer_0/rnn/while/Lesshidden_layer_0/rnn/while/Less_1
R
!hidden_layer_0/rnn/while/LoopCondLoopCond#hidden_layer_0/rnn/while/LogicalAnd
�
hidden_layer_0/rnn/while/SwitchSwitchhidden_layer_0/rnn/while/Merge!hidden_layer_0/rnn/while/LoopCond*
T0*1
_class'
%#loc:@hidden_layer_0/rnn/while/Merge
�
!hidden_layer_0/rnn/while/Switch_1Switch hidden_layer_0/rnn/while/Merge_1!hidden_layer_0/rnn/while/LoopCond*
T0*3
_class)
'%loc:@hidden_layer_0/rnn/while/Merge_1
�
!hidden_layer_0/rnn/while/Switch_2Switch hidden_layer_0/rnn/while/Merge_2!hidden_layer_0/rnn/while/LoopCond*
T0*3
_class)
'%loc:@hidden_layer_0/rnn/while/Merge_2
�
!hidden_layer_0/rnn/while/Switch_3Switch hidden_layer_0/rnn/while/Merge_3!hidden_layer_0/rnn/while/LoopCond*
T0*3
_class)
'%loc:@hidden_layer_0/rnn/while/Merge_3
�
!hidden_layer_0/rnn/while/Switch_4Switch hidden_layer_0/rnn/while/Merge_4!hidden_layer_0/rnn/while/LoopCond*
T0*3
_class)
'%loc:@hidden_layer_0/rnn/while/Merge_4
Y
!hidden_layer_0/rnn/while/IdentityIdentity!hidden_layer_0/rnn/while/Switch:1*
T0
]
#hidden_layer_0/rnn/while/Identity_1Identity#hidden_layer_0/rnn/while/Switch_1:1*
T0
]
#hidden_layer_0/rnn/while/Identity_2Identity#hidden_layer_0/rnn/while/Switch_2:1*
T0
]
#hidden_layer_0/rnn/while/Identity_3Identity#hidden_layer_0/rnn/while/Switch_3:1*
T0
]
#hidden_layer_0/rnn/while/Identity_4Identity#hidden_layer_0/rnn/while/Switch_4:1*
T0
l
hidden_layer_0/rnn/while/add/yConst"^hidden_layer_0/rnn/while/Identity*
dtype0*
value	B :
q
hidden_layer_0/rnn/while/addAddV2!hidden_layer_0/rnn/while/Identityhidden_layer_0/rnn/while/add/y*
T0
�
*hidden_layer_0/rnn/while/TensorArrayReadV3TensorArrayReadV30hidden_layer_0/rnn/while/TensorArrayReadV3/Enter#hidden_layer_0/rnn/while/Identity_12hidden_layer_0/rnn/while/TensorArrayReadV3/Enter_1*
dtype0
�
0hidden_layer_0/rnn/while/TensorArrayReadV3/EnterEnter hidden_layer_0/rnn/TensorArray_1*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
2hidden_layer_0/rnn/while/TensorArrayReadV3/Enter_1EnterMhidden_layer_0/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
%hidden_layer_0/rnn/while/GreaterEqualGreaterEqual#hidden_layer_0/rnn/while/Identity_1+hidden_layer_0/rnn/while/GreaterEqual/Enter*
T0
�
+hidden_layer_0/rnn/while/GreaterEqual/EnterEnterhidden_layer_0/rnn/CheckSeqLen*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
Dhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/shapeConst*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
dtype0*
valueB"�  �  
�
Bhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/minConst*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
dtype0*
valueB
 *�
�
Bhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/maxConst*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
dtype0*
valueB
 *=
�
Lhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformRandomUniformDhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/shape*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
dtype0*

seed *
seed2 
�
Bhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/subSubBhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/maxBhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel
�
Bhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/mulMulLhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/RandomUniformBhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/sub*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel
�
>hidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniformAddV2Bhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/mulBhidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform/min*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel
�
#hidden_layer_0/rnn/lstm_cell/kernel
VariableV2*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
	container *
dtype0*
shape:
��*
shared_name 
�
*hidden_layer_0/rnn/lstm_cell/kernel/AssignAssign#hidden_layer_0/rnn/lstm_cell/kernel>hidden_layer_0/rnn/lstm_cell/kernel/Initializer/random_uniform*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
use_locking(*
validate_shape(
b
(hidden_layer_0/rnn/lstm_cell/kernel/readIdentity#hidden_layer_0/rnn/lstm_cell/kernel*
T0
�
3hidden_layer_0/rnn/lstm_cell/bias/Initializer/zerosConst*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
dtype0*
valueB�*    
�
!hidden_layer_0/rnn/lstm_cell/bias
VariableV2*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
	container *
dtype0*
shape:�*
shared_name 
�
(hidden_layer_0/rnn/lstm_cell/bias/AssignAssign!hidden_layer_0/rnn/lstm_cell/bias3hidden_layer_0/rnn/lstm_cell/bias/Initializer/zeros*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
^
&hidden_layer_0/rnn/lstm_cell/bias/readIdentity!hidden_layer_0/rnn/lstm_cell/bias*
T0
|
.hidden_layer_0/rnn/while/lstm_cell/concat/axisConst"^hidden_layer_0/rnn/while/Identity*
dtype0*
value	B :
�
)hidden_layer_0/rnn/while/lstm_cell/concatConcatV2*hidden_layer_0/rnn/while/TensorArrayReadV3#hidden_layer_0/rnn/while/Identity_4.hidden_layer_0/rnn/while/lstm_cell/concat/axis*
N*
T0*

Tidx0
�
)hidden_layer_0/rnn/while/lstm_cell/MatMulMatMul)hidden_layer_0/rnn/while/lstm_cell/concat/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter*
T0*
transpose_a( *
transpose_b( 
�
/hidden_layer_0/rnn/while/lstm_cell/MatMul/EnterEnter(hidden_layer_0/rnn/lstm_cell/kernel/read*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
*hidden_layer_0/rnn/while/lstm_cell/BiasAddBiasAdd)hidden_layer_0/rnn/while/lstm_cell/MatMul0hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC
�
0hidden_layer_0/rnn/while/lstm_cell/BiasAdd/EnterEnter&hidden_layer_0/rnn/lstm_cell/bias/read*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
2hidden_layer_0/rnn/while/lstm_cell/split/split_dimConst"^hidden_layer_0/rnn/while/Identity*
dtype0*
value	B :
�
(hidden_layer_0/rnn/while/lstm_cell/splitSplit2hidden_layer_0/rnn/while/lstm_cell/split/split_dim*hidden_layer_0/rnn/while/lstm_cell/BiasAdd*
T0*
	num_split
y
(hidden_layer_0/rnn/while/lstm_cell/add/yConst"^hidden_layer_0/rnn/while/Identity*
dtype0*
valueB
 *  �?
�
&hidden_layer_0/rnn/while/lstm_cell/addAddV2*hidden_layer_0/rnn/while/lstm_cell/split:2(hidden_layer_0/rnn/while/lstm_cell/add/y*
T0
f
*hidden_layer_0/rnn/while/lstm_cell/SigmoidSigmoid&hidden_layer_0/rnn/while/lstm_cell/add*
T0
�
&hidden_layer_0/rnn/while/lstm_cell/mulMul*hidden_layer_0/rnn/while/lstm_cell/Sigmoid#hidden_layer_0/rnn/while/Identity_3*
T0
j
,hidden_layer_0/rnn/while/lstm_cell/Sigmoid_1Sigmoid(hidden_layer_0/rnn/while/lstm_cell/split*
T0
d
'hidden_layer_0/rnn/while/lstm_cell/TanhTanh*hidden_layer_0/rnn/while/lstm_cell/split:1*
T0
�
(hidden_layer_0/rnn/while/lstm_cell/mul_1Mul,hidden_layer_0/rnn/while/lstm_cell/Sigmoid_1'hidden_layer_0/rnn/while/lstm_cell/Tanh*
T0
�
(hidden_layer_0/rnn/while/lstm_cell/add_1AddV2&hidden_layer_0/rnn/while/lstm_cell/mul(hidden_layer_0/rnn/while/lstm_cell/mul_1*
T0
l
,hidden_layer_0/rnn/while/lstm_cell/Sigmoid_2Sigmoid*hidden_layer_0/rnn/while/lstm_cell/split:3*
T0
d
)hidden_layer_0/rnn/while/lstm_cell/Tanh_1Tanh(hidden_layer_0/rnn/while/lstm_cell/add_1*
T0
�
(hidden_layer_0/rnn/while/lstm_cell/mul_2Mul,hidden_layer_0/rnn/while/lstm_cell/Sigmoid_2)hidden_layer_0/rnn/while/lstm_cell/Tanh_1*
T0
o
hidden_layer_0/rnn/while/sub/xConst"^hidden_layer_0/rnn/while/Identity*
dtype0*
valueB
 *  �?
p
hidden_layer_0/rnn/while/subSubhidden_layer_0/rnn/while/sub/x"hidden_layer_0/rnn/while/sub/Enter*
T0
�
"hidden_layer_0/rnn/while/sub/EnterEnter	keep_prob*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
w
&hidden_layer_0/rnn/while/dropout/ConstConst"^hidden_layer_0/rnn/while/Identity*
dtype0*
valueB
 *  �?
z
$hidden_layer_0/rnn/while/dropout/SubSub&hidden_layer_0/rnn/while/dropout/Consthidden_layer_0/rnn/while/sub*
T0
�
(hidden_layer_0/rnn/while/dropout/RealDivRealDiv(hidden_layer_0/rnn/while/lstm_cell/mul_2$hidden_layer_0/rnn/while/dropout/Sub*
T0
r
&hidden_layer_0/rnn/while/dropout/ShapeShape(hidden_layer_0/rnn/while/lstm_cell/mul_2*
T0*
out_type0
�
=hidden_layer_0/rnn/while/dropout/random_uniform/RandomUniformRandomUniform&hidden_layer_0/rnn/while/dropout/Shape*
T0*
dtype0*

seed *
seed2 
�
-hidden_layer_0/rnn/while/dropout/GreaterEqualGreaterEqual=hidden_layer_0/rnn/while/dropout/random_uniform/RandomUniformhidden_layer_0/rnn/while/sub*
T0
�
%hidden_layer_0/rnn/while/dropout/CastCast-hidden_layer_0/rnn/while/dropout/GreaterEqual*

DstT0*

SrcT0
*
Truncate( 
�
$hidden_layer_0/rnn/while/dropout/MulMul(hidden_layer_0/rnn/while/dropout/RealDiv%hidden_layer_0/rnn/while/dropout/Cast*
T0
�
hidden_layer_0/rnn/while/SelectSelect%hidden_layer_0/rnn/while/GreaterEqual%hidden_layer_0/rnn/while/Select/Enter$hidden_layer_0/rnn/while/dropout/Mul*
T0*7
_class-
+)loc:@hidden_layer_0/rnn/while/dropout/Mul
�
%hidden_layer_0/rnn/while/Select/EnterEnterhidden_layer_0/rnn/zeros*
T0*7
_class-
+)loc:@hidden_layer_0/rnn/while/dropout/Mul*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
!hidden_layer_0/rnn/while/Select_1Select%hidden_layer_0/rnn/while/GreaterEqual#hidden_layer_0/rnn/while/Identity_3(hidden_layer_0/rnn/while/lstm_cell/add_1*
T0*;
_class1
/-loc:@hidden_layer_0/rnn/while/lstm_cell/add_1
�
!hidden_layer_0/rnn/while/Select_2Select%hidden_layer_0/rnn/while/GreaterEqual#hidden_layer_0/rnn/while/Identity_4(hidden_layer_0/rnn/while/lstm_cell/mul_2*
T0*;
_class1
/-loc:@hidden_layer_0/rnn/while/lstm_cell/mul_2
�
<hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Bhidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter#hidden_layer_0/rnn/while/Identity_1hidden_layer_0/rnn/while/Select#hidden_layer_0/rnn/while/Identity_2*
T0*7
_class-
+)loc:@hidden_layer_0/rnn/while/dropout/Mul
�
Bhidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterhidden_layer_0/rnn/TensorArray*
T0*7
_class-
+)loc:@hidden_layer_0/rnn/while/dropout/Mul*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
n
 hidden_layer_0/rnn/while/add_1/yConst"^hidden_layer_0/rnn/while/Identity*
dtype0*
value	B :
w
hidden_layer_0/rnn/while/add_1AddV2#hidden_layer_0/rnn/while/Identity_1 hidden_layer_0/rnn/while/add_1/y*
T0
^
&hidden_layer_0/rnn/while/NextIterationNextIterationhidden_layer_0/rnn/while/add*
T0
b
(hidden_layer_0/rnn/while/NextIteration_1NextIterationhidden_layer_0/rnn/while/add_1*
T0
�
(hidden_layer_0/rnn/while/NextIteration_2NextIteration<hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0
e
(hidden_layer_0/rnn/while/NextIteration_3NextIteration!hidden_layer_0/rnn/while/Select_1*
T0
e
(hidden_layer_0/rnn/while/NextIteration_4NextIteration!hidden_layer_0/rnn/while/Select_2*
T0
O
hidden_layer_0/rnn/while/ExitExithidden_layer_0/rnn/while/Switch*
T0
S
hidden_layer_0/rnn/while/Exit_1Exit!hidden_layer_0/rnn/while/Switch_1*
T0
S
hidden_layer_0/rnn/while/Exit_2Exit!hidden_layer_0/rnn/while/Switch_2*
T0
S
hidden_layer_0/rnn/while/Exit_3Exit!hidden_layer_0/rnn/while/Switch_3*
T0
S
hidden_layer_0/rnn/while/Exit_4Exit!hidden_layer_0/rnn/while/Switch_4*
T0
�
/hidden_layer_0/rnn/TensorArrayStack/range/startConst*1
_class'
%#loc:@hidden_layer_0/rnn/TensorArray*
dtype0*
value	B : 
�
/hidden_layer_0/rnn/TensorArrayStack/range/deltaConst*1
_class'
%#loc:@hidden_layer_0/rnn/TensorArray*
dtype0*
value	B :
�
)hidden_layer_0/rnn/TensorArrayStack/rangeRange/hidden_layer_0/rnn/TensorArrayStack/range/start"hidden_layer_0/rnn/strided_slice_1/hidden_layer_0/rnn/TensorArrayStack/range/delta*

Tidx0*1
_class'
%#loc:@hidden_layer_0/rnn/TensorArray
�
7hidden_layer_0/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3hidden_layer_0/rnn/TensorArray)hidden_layer_0/rnn/TensorArrayStack/rangehidden_layer_0/rnn/while/Exit_2*1
_class'
%#loc:@hidden_layer_0/rnn/TensorArray*
dtype0*$
element_shape:���������f
H
hidden_layer_0/rnn/Const_4Const*
dtype0*
valueB:f
C
hidden_layer_0/rnn/Rank_1Const*
dtype0*
value	B :
J
 hidden_layer_0/rnn/range_1/startConst*
dtype0*
value	B :
J
 hidden_layer_0/rnn/range_1/deltaConst*
dtype0*
value	B :
�
hidden_layer_0/rnn/range_1Range hidden_layer_0/rnn/range_1/starthidden_layer_0/rnn/Rank_1 hidden_layer_0/rnn/range_1/delta*

Tidx0
Y
$hidden_layer_0/rnn/concat_2/values_0Const*
dtype0*
valueB"       
J
 hidden_layer_0/rnn/concat_2/axisConst*
dtype0*
value	B : 
�
hidden_layer_0/rnn/concat_2ConcatV2$hidden_layer_0/rnn/concat_2/values_0hidden_layer_0/rnn/range_1 hidden_layer_0/rnn/concat_2/axis*
N*
T0*

Tidx0
�
hidden_layer_0/rnn/transpose_1	Transpose7hidden_layer_0/rnn/TensorArrayStack/TensorArrayGatherV3hidden_layer_0/rnn/concat_2*
T0*
Tperm0
�
4output_layer/weights/Initializer/random_normal/shapeConst*'
_class
loc:@output_layer/weights*
dtype0*
valueB"f   I  
�
3output_layer/weights/Initializer/random_normal/meanConst*'
_class
loc:@output_layer/weights*
dtype0*
valueB
 *    
�
5output_layer/weights/Initializer/random_normal/stddevConst*'
_class
loc:@output_layer/weights*
dtype0*
valueB
 *R��<
�
Coutput_layer/weights/Initializer/random_normal/RandomStandardNormalRandomStandardNormal4output_layer/weights/Initializer/random_normal/shape*
T0*'
_class
loc:@output_layer/weights*
dtype0*

seed *
seed2 
�
2output_layer/weights/Initializer/random_normal/mulMulCoutput_layer/weights/Initializer/random_normal/RandomStandardNormal5output_layer/weights/Initializer/random_normal/stddev*
T0*'
_class
loc:@output_layer/weights
�
.output_layer/weights/Initializer/random_normalAddV22output_layer/weights/Initializer/random_normal/mul3output_layer/weights/Initializer/random_normal/mean*
T0*'
_class
loc:@output_layer/weights
�
output_layer/weights
VariableV2*'
_class
loc:@output_layer/weights*
	container *
dtype0*
shape:	f�*
shared_name 
�
output_layer/weights/AssignAssignoutput_layer/weights.output_layer/weights/Initializer/random_normal*
T0*'
_class
loc:@output_layer/weights*
use_locking(*
validate_shape(
m
output_layer/weights/readIdentityoutput_layer/weights*
T0*'
_class
loc:@output_layer/weights
�
3output_layer/biases/Initializer/random_normal/shapeConst*&
_class
loc:@output_layer/biases*
dtype0*
valueB:�
�
2output_layer/biases/Initializer/random_normal/meanConst*&
_class
loc:@output_layer/biases*
dtype0*
valueB
 *    
�
4output_layer/biases/Initializer/random_normal/stddevConst*&
_class
loc:@output_layer/biases*
dtype0*
valueB
 *R��<
�
Boutput_layer/biases/Initializer/random_normal/RandomStandardNormalRandomStandardNormal3output_layer/biases/Initializer/random_normal/shape*
T0*&
_class
loc:@output_layer/biases*
dtype0*

seed *
seed2 
�
1output_layer/biases/Initializer/random_normal/mulMulBoutput_layer/biases/Initializer/random_normal/RandomStandardNormal4output_layer/biases/Initializer/random_normal/stddev*
T0*&
_class
loc:@output_layer/biases
�
-output_layer/biases/Initializer/random_normalAddV21output_layer/biases/Initializer/random_normal/mul2output_layer/biases/Initializer/random_normal/mean*
T0*&
_class
loc:@output_layer/biases
�
output_layer/biases
VariableV2*&
_class
loc:@output_layer/biases*
	container *
dtype0*
shape:�*
shared_name 
�
output_layer/biases/AssignAssignoutput_layer/biases-output_layer/biases/Initializer/random_normal*
T0*&
_class
loc:@output_layer/biases*
use_locking(*
validate_shape(
j
output_layer/biases/readIdentityoutput_layer/biases*
T0*&
_class
loc:@output_layer/biases
T
output_layer/ShapeShapehidden_layer_0/rnn/transpose_1*
T0*
out_type0
N
 output_layer/strided_slice/stackConst*
dtype0*
valueB:
P
"output_layer/strided_slice/stack_1Const*
dtype0*
valueB:
P
"output_layer/strided_slice/stack_2Const*
dtype0*
valueB:
�
output_layer/strided_sliceStridedSliceoutput_layer/Shape output_layer/strided_slice/stack"output_layer/strided_slice/stack_1"output_layer/strided_slice/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
O
output_layer/Reshape/shapeConst*
dtype0*
valueB"����f   
r
output_layer/ReshapeReshapehidden_layer_0/rnn/transpose_1output_layer/Reshape/shape*
T0*
Tshape0
}
output_layer/MatMulMatMuloutput_layer/Reshapeoutput_layer/weights/read*
T0*
transpose_a( *
transpose_b( 
Q
output_layer/addAddV2output_layer/MatMuloutput_layer/biases/read*
T0
Q
output_layer/Reshape_1/shape/0Const*
dtype0*
valueB :
���������
I
output_layer/Reshape_1/shape/2Const*
dtype0*
value
B :�
�
output_layer/Reshape_1/shapePackoutput_layer/Reshape_1/shape/0output_layer/strided_sliceoutput_layer/Reshape_1/shape/2*
N*
T0*

axis 
h
output_layer/Reshape_1Reshapeoutput_layer/addoutput_layer/Reshape_1/shape*
T0*
Tshape0
>
output_layer/predsSigmoidoutput_layer/Reshape_1*
T0
D
output_layer/NotEqual/yConst*
dtype0*
valueB
 *    
j
output_layer/NotEqualNotEqualy_seqoutput_layer/NotEqual/y*
T0*
incompatible_shape_error(
;
output_layer/WhereWhereoutput_layer/NotEqual*
T0

l
output_layer/GatherNdGatherNdoutput_layer/Reshape_1output_layer/Where*
Tindices0	*
Tparams0
j
output_layer/GatherNd_1GatherNdoutput_layer/predsoutput_layer/Where*
Tindices0	*
Tparams0
^
output_layer/GatherNd_2GatherNdy_corroutput_layer/Where*
Tindices0	*
Tparams0
R
%output_layer/logistic_loss/zeros_like	ZerosLikeoutput_layer/GatherNd*
T0
~
'output_layer/logistic_loss/GreaterEqualGreaterEqualoutput_layer/GatherNd%output_layer/logistic_loss/zeros_like*
T0
�
!output_layer/logistic_loss/SelectSelect'output_layer/logistic_loss/GreaterEqualoutput_layer/GatherNd%output_layer/logistic_loss/zeros_like*
T0
E
output_layer/logistic_loss/NegNegoutput_layer/GatherNd*
T0
�
#output_layer/logistic_loss/Select_1Select'output_layer/logistic_loss/GreaterEqualoutput_layer/logistic_loss/Negoutput_layer/GatherNd*
T0
^
output_layer/logistic_loss/mulMuloutput_layer/GatherNdoutput_layer/GatherNd_2*
T0
q
output_layer/logistic_loss/subSub!output_layer/logistic_loss/Selectoutput_layer/logistic_loss/mul*
T0
S
output_layer/logistic_loss/ExpExp#output_layer/logistic_loss/Select_1*
T0
R
 output_layer/logistic_loss/Log1pLog1poutput_layer/logistic_loss/Exp*
T0
n
output_layer/logistic_lossAddV2output_layer/logistic_loss/sub output_layer/logistic_loss/Log1p*
T0
@
output_layer/ConstConst*
dtype0*
valueB: 
o
output_layer/MeanMeanoutput_layer/logistic_lossoutput_layer/Const*
T0*

Tidx0*
	keep_dims( 
[
"output_layer/strided_slice_1/stackConst*
dtype0*!
valueB"            
]
$output_layer/strided_slice_1/stack_1Const*
dtype0*!
valueB"        I  
]
$output_layer/strided_slice_1/stack_2Const*
dtype0*!
valueB"         
�
output_layer/strided_slice_1StridedSliceX"output_layer/strided_slice_1/stack$output_layer/strided_slice_1/stack_1$output_layer/strided_slice_1/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
[
"output_layer/strided_slice_2/stackConst*
dtype0*!
valueB"        I  
]
$output_layer/strided_slice_2/stack_1Const*
dtype0*!
valueB"            
]
$output_layer/strided_slice_2/stack_2Const*
dtype0*!
valueB"         
�
output_layer/strided_slice_2StridedSliceX"output_layer/strided_slice_2/stack$output_layer/strided_slice_2/stack_1$output_layer/strided_slice_2/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
F
output_layer/NotEqual_1/yConst*
dtype0*
valueB
 *    
�
output_layer/NotEqual_1NotEqualoutput_layer/strided_slice_1output_layer/NotEqual_1/y*
T0*
incompatible_shape_error(
?
output_layer/Where_1Whereoutput_layer/NotEqual_1*
T0

p
output_layer/GatherNd_3GatherNdoutput_layer/Reshape_1output_layer/Where_1*
Tindices0	*
Tparams0
l
output_layer/GatherNd_4GatherNdoutput_layer/predsoutput_layer/Where_1*
Tindices0	*
Tparams0
v
output_layer/GatherNd_5GatherNdoutput_layer/strided_slice_2output_layer/Where_1*
Tindices0	*
Tparams0
V
'output_layer/logistic_loss_1/zeros_like	ZerosLikeoutput_layer/GatherNd_3*
T0
�
)output_layer/logistic_loss_1/GreaterEqualGreaterEqualoutput_layer/GatherNd_3'output_layer/logistic_loss_1/zeros_like*
T0
�
#output_layer/logistic_loss_1/SelectSelect)output_layer/logistic_loss_1/GreaterEqualoutput_layer/GatherNd_3'output_layer/logistic_loss_1/zeros_like*
T0
I
 output_layer/logistic_loss_1/NegNegoutput_layer/GatherNd_3*
T0
�
%output_layer/logistic_loss_1/Select_1Select)output_layer/logistic_loss_1/GreaterEqual output_layer/logistic_loss_1/Negoutput_layer/GatherNd_3*
T0
b
 output_layer/logistic_loss_1/mulMuloutput_layer/GatherNd_3output_layer/GatherNd_5*
T0
w
 output_layer/logistic_loss_1/subSub#output_layer/logistic_loss_1/Select output_layer/logistic_loss_1/mul*
T0
W
 output_layer/logistic_loss_1/ExpExp%output_layer/logistic_loss_1/Select_1*
T0
V
"output_layer/logistic_loss_1/Log1pLog1p output_layer/logistic_loss_1/Exp*
T0
t
output_layer/logistic_loss_1AddV2 output_layer/logistic_loss_1/sub"output_layer/logistic_loss_1/Log1p*
T0
B
output_layer/Const_1Const*
dtype0*
valueB: 
u
output_layer/Mean_1Meanoutput_layer/logistic_loss_1output_layer/Const_1*
T0*

Tidx0*
	keep_dims( 
?
output_layer/mul/xConst*
dtype0*
valueB
 *���=
I
output_layer/mulMuloutput_layer/mul/xoutput_layer/Mean_1*
T0
I
output_layer/add_1AddV2output_layer/Meanoutput_layer/mul*
T0
'
output_layer/AbsAbsy_seq*
T0
L
"output_layer/Max/reduction_indicesConst*
dtype0*
value	B :
s
output_layer/MaxMaxoutput_layer/Abs"output_layer/Max/reduction_indices*
T0*

Tidx0*
	keep_dims( 
4
output_layer/SignSignoutput_layer/Max*
T0
L
"output_layer/Sum/reduction_indicesConst*
dtype0*
value	B :
t
output_layer/SumSumoutput_layer/Sign"output_layer/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
S
output_layer/CastCastoutput_layer/Sum*

DstT0*

SrcT0*
Truncate( 
V
output_layer/Cast_1Castoutput_layer/Cast*

DstT0*

SrcT0*
Truncate( 
B
output_layer/Const_2Const*
dtype0*
valueB: 
j
output_layer/Sum_1Sumoutput_layer/Cast_1output_layer/Const_2*
T0*

Tidx0*
	keep_dims( 
[
"output_layer/strided_slice_3/stackConst*
dtype0*!
valueB"           
]
$output_layer/strided_slice_3/stack_1Const*
dtype0*!
valueB"            
]
$output_layer/strided_slice_3/stack_2Const*
dtype0*!
valueB"         
�
output_layer/strided_slice_3StridedSliceoutput_layer/preds"output_layer/strided_slice_3/stack$output_layer/strided_slice_3/stack_1$output_layer/strided_slice_3/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
[
"output_layer/strided_slice_4/stackConst*
dtype0*!
valueB"            
]
$output_layer/strided_slice_4/stack_1Const*
dtype0*!
valueB"    ����    
]
$output_layer/strided_slice_4/stack_2Const*
dtype0*!
valueB"         
�
output_layer/strided_slice_4StridedSliceoutput_layer/preds"output_layer/strided_slice_4/stack$output_layer/strided_slice_4/stack_1$output_layer/strided_slice_4/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
\
output_layer/subSuboutput_layer/strided_slice_3output_layer/strided_slice_4*
T0
4
output_layer/Abs_1Absoutput_layer/sub*
T0
M
output_layer/Const_3Const*
dtype0*!
valueB"          
i
output_layer/Sum_2Sumoutput_layer/Abs_1output_layer/Const_3*
T0*

Tidx0*
	keep_dims( 
P
output_layer/truedivRealDivoutput_layer/Sum_2output_layer/Sum_1*
T0
E
output_layer/truediv_1/yConst*
dtype0*
valueB
 *  �D
Z
output_layer/truediv_1RealDivoutput_layer/truedivoutput_layer/truediv_1/y*
T0
A
output_layer/mul_1/xConst*
dtype0*
valueB
 *���<
P
output_layer/mul_1Muloutput_layer/mul_1/xoutput_layer/truediv_1*
T0
L
output_layer/add_2AddV2output_layer/add_1output_layer/mul_1*
T0
[
"output_layer/strided_slice_5/stackConst*
dtype0*!
valueB"           
]
$output_layer/strided_slice_5/stack_1Const*
dtype0*!
valueB"            
]
$output_layer/strided_slice_5/stack_2Const*
dtype0*!
valueB"         
�
output_layer/strided_slice_5StridedSliceoutput_layer/preds"output_layer/strided_slice_5/stack$output_layer/strided_slice_5/stack_1$output_layer/strided_slice_5/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
[
"output_layer/strided_slice_6/stackConst*
dtype0*!
valueB"            
]
$output_layer/strided_slice_6/stack_1Const*
dtype0*!
valueB"    ����    
]
$output_layer/strided_slice_6/stack_2Const*
dtype0*!
valueB"         
�
output_layer/strided_slice_6StridedSliceoutput_layer/preds"output_layer/strided_slice_6/stack$output_layer/strided_slice_6/stack_1$output_layer/strided_slice_6/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
^
output_layer/sub_1Suboutput_layer/strided_slice_5output_layer/strided_slice_6*
T0
:
output_layer/SquareSquareoutput_layer/sub_1*
T0
M
output_layer/Const_4Const*
dtype0*!
valueB"          
j
output_layer/Sum_3Sumoutput_layer/Squareoutput_layer/Const_4*
T0*

Tidx0*
	keep_dims( 
R
output_layer/truediv_2RealDivoutput_layer/Sum_3output_layer/Sum_1*
T0
E
output_layer/truediv_3/yConst*
dtype0*
valueB
 *  �D
\
output_layer/truediv_3RealDivoutput_layer/truediv_2output_layer/truediv_3/y*
T0
A
output_layer/mul_2/xConst*
dtype0*
valueB
 *  @@
P
output_layer/mul_2Muloutput_layer/mul_2/xoutput_layer/truediv_3*
T0
L
output_layer/add_3AddV2output_layer/add_2output_layer/mul_2*
T0
B
Optimizer/gradients/ShapeConst*
dtype0*
valueB 
P
#Optimizer/gradients/grad_ys_0/ConstConst*
dtype0*
valueB
 *  �?
�
Optimizer/gradients/grad_ys_0FillOptimizer/gradients/Shape#Optimizer/gradients/grad_ys_0/Const*
T0*

index_type0
E
Optimizer/gradients/f_countConst*
dtype0*
value	B : 
�
Optimizer/gradients/f_count_1EnterOptimizer/gradients/f_count*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
v
Optimizer/gradients/MergeMergeOptimizer/gradients/f_count_1!Optimizer/gradients/NextIteration*
N*
T0
k
Optimizer/gradients/SwitchSwitchOptimizer/gradients/Merge!hidden_layer_0/rnn/while/LoopCond*
T0
g
Optimizer/gradients/Add/yConst"^hidden_layer_0/rnn/while/Identity*
dtype0*
value	B :
b
Optimizer/gradients/AddAddV2Optimizer/gradients/Switch:1Optimizer/gradients/Add/y*
T0
�
!Optimizer/gradients/NextIterationNextIterationOptimizer/gradients/AddN^Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPushV2R^Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/StackPushV2R^Optimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/StackPushV2t^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2k^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPushV2`^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPushV2b^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPushV2_1N^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/StackPushV2P^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/StackPushV2d^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPushV2R^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/StackPushV2V^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/StackPushV2X^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2d^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2f^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1b^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2d^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2_1V^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2d^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2f^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1R^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2T^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2d^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2f^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1R^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2T^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2b^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2d^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1R^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
T0
J
Optimizer/gradients/f_count_2ExitOptimizer/gradients/Switch*
T0
E
Optimizer/gradients/b_countConst*
dtype0*
value	B :
�
Optimizer/gradients/b_count_1EnterOptimizer/gradients/f_count_2*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
z
Optimizer/gradients/Merge_1MergeOptimizer/gradients/b_count_1#Optimizer/gradients/NextIteration_1*
N*
T0
~
 Optimizer/gradients/GreaterEqualGreaterEqualOptimizer/gradients/Merge_1&Optimizer/gradients/GreaterEqual/Enter*
T0
�
&Optimizer/gradients/GreaterEqual/EnterEnterOptimizer/gradients/b_count*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
K
Optimizer/gradients/b_count_2LoopCond Optimizer/gradients/GreaterEqual
k
Optimizer/gradients/Switch_1SwitchOptimizer/gradients/Merge_1Optimizer/gradients/b_count_2*
T0
o
Optimizer/gradients/SubSubOptimizer/gradients/Switch_1:1&Optimizer/gradients/GreaterEqual/Enter*
T0
�
#Optimizer/gradients/NextIteration_1NextIterationOptimizer/gradients/Subf^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/b_sync*
T0
L
Optimizer/gradients/b_count_3ExitOptimizer/gradients/Switch_1*
T0
d
<Optimizer/gradients/output_layer/add_3_grad/tuple/group_depsNoOp^Optimizer/gradients/grad_ys_0
�
DOptimizer/gradients/output_layer/add_3_grad/tuple/control_dependencyIdentityOptimizer/gradients/grad_ys_0=^Optimizer/gradients/output_layer/add_3_grad/tuple/group_deps*
T0*0
_class&
$"loc:@Optimizer/gradients/grad_ys_0
�
FOptimizer/gradients/output_layer/add_3_grad/tuple/control_dependency_1IdentityOptimizer/gradients/grad_ys_0=^Optimizer/gradients/output_layer/add_3_grad/tuple/group_deps*
T0*0
_class&
$"loc:@Optimizer/gradients/grad_ys_0
�
<Optimizer/gradients/output_layer/add_2_grad/tuple/group_depsNoOpE^Optimizer/gradients/output_layer/add_3_grad/tuple/control_dependency
�
DOptimizer/gradients/output_layer/add_2_grad/tuple/control_dependencyIdentityDOptimizer/gradients/output_layer/add_3_grad/tuple/control_dependency=^Optimizer/gradients/output_layer/add_2_grad/tuple/group_deps*
T0*0
_class&
$"loc:@Optimizer/gradients/grad_ys_0
�
FOptimizer/gradients/output_layer/add_2_grad/tuple/control_dependency_1IdentityDOptimizer/gradients/output_layer/add_3_grad/tuple/control_dependency=^Optimizer/gradients/output_layer/add_2_grad/tuple/group_deps*
T0*0
_class&
$"loc:@Optimizer/gradients/grad_ys_0
�
/Optimizer/gradients/output_layer/mul_2_grad/MulMulFOptimizer/gradients/output_layer/add_3_grad/tuple/control_dependency_1output_layer/truediv_3*
T0
�
1Optimizer/gradients/output_layer/mul_2_grad/Mul_1MulFOptimizer/gradients/output_layer/add_3_grad/tuple/control_dependency_1output_layer/mul_2/x*
T0
�
<Optimizer/gradients/output_layer/mul_2_grad/tuple/group_depsNoOp0^Optimizer/gradients/output_layer/mul_2_grad/Mul2^Optimizer/gradients/output_layer/mul_2_grad/Mul_1
�
DOptimizer/gradients/output_layer/mul_2_grad/tuple/control_dependencyIdentity/Optimizer/gradients/output_layer/mul_2_grad/Mul=^Optimizer/gradients/output_layer/mul_2_grad/tuple/group_deps*
T0*B
_class8
64loc:@Optimizer/gradients/output_layer/mul_2_grad/Mul
�
FOptimizer/gradients/output_layer/mul_2_grad/tuple/control_dependency_1Identity1Optimizer/gradients/output_layer/mul_2_grad/Mul_1=^Optimizer/gradients/output_layer/mul_2_grad/tuple/group_deps*
T0*D
_class:
86loc:@Optimizer/gradients/output_layer/mul_2_grad/Mul_1
�
<Optimizer/gradients/output_layer/add_1_grad/tuple/group_depsNoOpE^Optimizer/gradients/output_layer/add_2_grad/tuple/control_dependency
�
DOptimizer/gradients/output_layer/add_1_grad/tuple/control_dependencyIdentityDOptimizer/gradients/output_layer/add_2_grad/tuple/control_dependency=^Optimizer/gradients/output_layer/add_1_grad/tuple/group_deps*
T0*0
_class&
$"loc:@Optimizer/gradients/grad_ys_0
�
FOptimizer/gradients/output_layer/add_1_grad/tuple/control_dependency_1IdentityDOptimizer/gradients/output_layer/add_2_grad/tuple/control_dependency=^Optimizer/gradients/output_layer/add_1_grad/tuple/group_deps*
T0*0
_class&
$"loc:@Optimizer/gradients/grad_ys_0
�
/Optimizer/gradients/output_layer/mul_1_grad/MulMulFOptimizer/gradients/output_layer/add_2_grad/tuple/control_dependency_1output_layer/truediv_1*
T0
�
1Optimizer/gradients/output_layer/mul_1_grad/Mul_1MulFOptimizer/gradients/output_layer/add_2_grad/tuple/control_dependency_1output_layer/mul_1/x*
T0
�
<Optimizer/gradients/output_layer/mul_1_grad/tuple/group_depsNoOp0^Optimizer/gradients/output_layer/mul_1_grad/Mul2^Optimizer/gradients/output_layer/mul_1_grad/Mul_1
�
DOptimizer/gradients/output_layer/mul_1_grad/tuple/control_dependencyIdentity/Optimizer/gradients/output_layer/mul_1_grad/Mul=^Optimizer/gradients/output_layer/mul_1_grad/tuple/group_deps*
T0*B
_class8
64loc:@Optimizer/gradients/output_layer/mul_1_grad/Mul
�
FOptimizer/gradients/output_layer/mul_1_grad/tuple/control_dependency_1Identity1Optimizer/gradients/output_layer/mul_1_grad/Mul_1=^Optimizer/gradients/output_layer/mul_1_grad/tuple/group_deps*
T0*D
_class:
86loc:@Optimizer/gradients/output_layer/mul_1_grad/Mul_1
^
5Optimizer/gradients/output_layer/truediv_3_grad/ShapeConst*
dtype0*
valueB 
`
7Optimizer/gradients/output_layer/truediv_3_grad/Shape_1Const*
dtype0*
valueB 
�
EOptimizer/gradients/output_layer/truediv_3_grad/BroadcastGradientArgsBroadcastGradientArgs5Optimizer/gradients/output_layer/truediv_3_grad/Shape7Optimizer/gradients/output_layer/truediv_3_grad/Shape_1*
T0
�
7Optimizer/gradients/output_layer/truediv_3_grad/RealDivRealDivFOptimizer/gradients/output_layer/mul_2_grad/tuple/control_dependency_1output_layer/truediv_3/y*
T0
�
3Optimizer/gradients/output_layer/truediv_3_grad/SumSum7Optimizer/gradients/output_layer/truediv_3_grad/RealDivEOptimizer/gradients/output_layer/truediv_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
7Optimizer/gradients/output_layer/truediv_3_grad/ReshapeReshape3Optimizer/gradients/output_layer/truediv_3_grad/Sum5Optimizer/gradients/output_layer/truediv_3_grad/Shape*
T0*
Tshape0
[
3Optimizer/gradients/output_layer/truediv_3_grad/NegNegoutput_layer/truediv_2*
T0
�
9Optimizer/gradients/output_layer/truediv_3_grad/RealDiv_1RealDiv3Optimizer/gradients/output_layer/truediv_3_grad/Negoutput_layer/truediv_3/y*
T0
�
9Optimizer/gradients/output_layer/truediv_3_grad/RealDiv_2RealDiv9Optimizer/gradients/output_layer/truediv_3_grad/RealDiv_1output_layer/truediv_3/y*
T0
�
3Optimizer/gradients/output_layer/truediv_3_grad/mulMulFOptimizer/gradients/output_layer/mul_2_grad/tuple/control_dependency_19Optimizer/gradients/output_layer/truediv_3_grad/RealDiv_2*
T0
�
5Optimizer/gradients/output_layer/truediv_3_grad/Sum_1Sum3Optimizer/gradients/output_layer/truediv_3_grad/mulGOptimizer/gradients/output_layer/truediv_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
9Optimizer/gradients/output_layer/truediv_3_grad/Reshape_1Reshape5Optimizer/gradients/output_layer/truediv_3_grad/Sum_17Optimizer/gradients/output_layer/truediv_3_grad/Shape_1*
T0*
Tshape0
�
@Optimizer/gradients/output_layer/truediv_3_grad/tuple/group_depsNoOp8^Optimizer/gradients/output_layer/truediv_3_grad/Reshape:^Optimizer/gradients/output_layer/truediv_3_grad/Reshape_1
�
HOptimizer/gradients/output_layer/truediv_3_grad/tuple/control_dependencyIdentity7Optimizer/gradients/output_layer/truediv_3_grad/ReshapeA^Optimizer/gradients/output_layer/truediv_3_grad/tuple/group_deps*
T0*J
_class@
><loc:@Optimizer/gradients/output_layer/truediv_3_grad/Reshape
�
JOptimizer/gradients/output_layer/truediv_3_grad/tuple/control_dependency_1Identity9Optimizer/gradients/output_layer/truediv_3_grad/Reshape_1A^Optimizer/gradients/output_layer/truediv_3_grad/tuple/group_deps*
T0*L
_classB
@>loc:@Optimizer/gradients/output_layer/truediv_3_grad/Reshape_1
f
8Optimizer/gradients/output_layer/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:
�
2Optimizer/gradients/output_layer/Mean_grad/ReshapeReshapeDOptimizer/gradients/output_layer/add_1_grad/tuple/control_dependency8Optimizer/gradients/output_layer/Mean_grad/Reshape/shape*
T0*
Tshape0
n
0Optimizer/gradients/output_layer/Mean_grad/ShapeShapeoutput_layer/logistic_loss*
T0*
out_type0
�
/Optimizer/gradients/output_layer/Mean_grad/TileTile2Optimizer/gradients/output_layer/Mean_grad/Reshape0Optimizer/gradients/output_layer/Mean_grad/Shape*
T0*

Tmultiples0
p
2Optimizer/gradients/output_layer/Mean_grad/Shape_1Shapeoutput_layer/logistic_loss*
T0*
out_type0
[
2Optimizer/gradients/output_layer/Mean_grad/Shape_2Const*
dtype0*
valueB 
^
0Optimizer/gradients/output_layer/Mean_grad/ConstConst*
dtype0*
valueB: 
�
/Optimizer/gradients/output_layer/Mean_grad/ProdProd2Optimizer/gradients/output_layer/Mean_grad/Shape_10Optimizer/gradients/output_layer/Mean_grad/Const*
T0*

Tidx0*
	keep_dims( 
`
2Optimizer/gradients/output_layer/Mean_grad/Const_1Const*
dtype0*
valueB: 
�
1Optimizer/gradients/output_layer/Mean_grad/Prod_1Prod2Optimizer/gradients/output_layer/Mean_grad/Shape_22Optimizer/gradients/output_layer/Mean_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
^
4Optimizer/gradients/output_layer/Mean_grad/Maximum/yConst*
dtype0*
value	B :
�
2Optimizer/gradients/output_layer/Mean_grad/MaximumMaximum1Optimizer/gradients/output_layer/Mean_grad/Prod_14Optimizer/gradients/output_layer/Mean_grad/Maximum/y*
T0
�
3Optimizer/gradients/output_layer/Mean_grad/floordivFloorDiv/Optimizer/gradients/output_layer/Mean_grad/Prod2Optimizer/gradients/output_layer/Mean_grad/Maximum*
T0
�
/Optimizer/gradients/output_layer/Mean_grad/CastCast3Optimizer/gradients/output_layer/Mean_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
�
2Optimizer/gradients/output_layer/Mean_grad/truedivRealDiv/Optimizer/gradients/output_layer/Mean_grad/Tile/Optimizer/gradients/output_layer/Mean_grad/Cast*
T0
�
-Optimizer/gradients/output_layer/mul_grad/MulMulFOptimizer/gradients/output_layer/add_1_grad/tuple/control_dependency_1output_layer/Mean_1*
T0
�
/Optimizer/gradients/output_layer/mul_grad/Mul_1MulFOptimizer/gradients/output_layer/add_1_grad/tuple/control_dependency_1output_layer/mul/x*
T0
�
:Optimizer/gradients/output_layer/mul_grad/tuple/group_depsNoOp.^Optimizer/gradients/output_layer/mul_grad/Mul0^Optimizer/gradients/output_layer/mul_grad/Mul_1
�
BOptimizer/gradients/output_layer/mul_grad/tuple/control_dependencyIdentity-Optimizer/gradients/output_layer/mul_grad/Mul;^Optimizer/gradients/output_layer/mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@Optimizer/gradients/output_layer/mul_grad/Mul
�
DOptimizer/gradients/output_layer/mul_grad/tuple/control_dependency_1Identity/Optimizer/gradients/output_layer/mul_grad/Mul_1;^Optimizer/gradients/output_layer/mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@Optimizer/gradients/output_layer/mul_grad/Mul_1
^
5Optimizer/gradients/output_layer/truediv_1_grad/ShapeConst*
dtype0*
valueB 
`
7Optimizer/gradients/output_layer/truediv_1_grad/Shape_1Const*
dtype0*
valueB 
�
EOptimizer/gradients/output_layer/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs5Optimizer/gradients/output_layer/truediv_1_grad/Shape7Optimizer/gradients/output_layer/truediv_1_grad/Shape_1*
T0
�
7Optimizer/gradients/output_layer/truediv_1_grad/RealDivRealDivFOptimizer/gradients/output_layer/mul_1_grad/tuple/control_dependency_1output_layer/truediv_1/y*
T0
�
3Optimizer/gradients/output_layer/truediv_1_grad/SumSum7Optimizer/gradients/output_layer/truediv_1_grad/RealDivEOptimizer/gradients/output_layer/truediv_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
7Optimizer/gradients/output_layer/truediv_1_grad/ReshapeReshape3Optimizer/gradients/output_layer/truediv_1_grad/Sum5Optimizer/gradients/output_layer/truediv_1_grad/Shape*
T0*
Tshape0
Y
3Optimizer/gradients/output_layer/truediv_1_grad/NegNegoutput_layer/truediv*
T0
�
9Optimizer/gradients/output_layer/truediv_1_grad/RealDiv_1RealDiv3Optimizer/gradients/output_layer/truediv_1_grad/Negoutput_layer/truediv_1/y*
T0
�
9Optimizer/gradients/output_layer/truediv_1_grad/RealDiv_2RealDiv9Optimizer/gradients/output_layer/truediv_1_grad/RealDiv_1output_layer/truediv_1/y*
T0
�
3Optimizer/gradients/output_layer/truediv_1_grad/mulMulFOptimizer/gradients/output_layer/mul_1_grad/tuple/control_dependency_19Optimizer/gradients/output_layer/truediv_1_grad/RealDiv_2*
T0
�
5Optimizer/gradients/output_layer/truediv_1_grad/Sum_1Sum3Optimizer/gradients/output_layer/truediv_1_grad/mulGOptimizer/gradients/output_layer/truediv_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
9Optimizer/gradients/output_layer/truediv_1_grad/Reshape_1Reshape5Optimizer/gradients/output_layer/truediv_1_grad/Sum_17Optimizer/gradients/output_layer/truediv_1_grad/Shape_1*
T0*
Tshape0
�
@Optimizer/gradients/output_layer/truediv_1_grad/tuple/group_depsNoOp8^Optimizer/gradients/output_layer/truediv_1_grad/Reshape:^Optimizer/gradients/output_layer/truediv_1_grad/Reshape_1
�
HOptimizer/gradients/output_layer/truediv_1_grad/tuple/control_dependencyIdentity7Optimizer/gradients/output_layer/truediv_1_grad/ReshapeA^Optimizer/gradients/output_layer/truediv_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@Optimizer/gradients/output_layer/truediv_1_grad/Reshape
�
JOptimizer/gradients/output_layer/truediv_1_grad/tuple/control_dependency_1Identity9Optimizer/gradients/output_layer/truediv_1_grad/Reshape_1A^Optimizer/gradients/output_layer/truediv_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@Optimizer/gradients/output_layer/truediv_1_grad/Reshape_1
^
5Optimizer/gradients/output_layer/truediv_2_grad/ShapeConst*
dtype0*
valueB 
`
7Optimizer/gradients/output_layer/truediv_2_grad/Shape_1Const*
dtype0*
valueB 
�
EOptimizer/gradients/output_layer/truediv_2_grad/BroadcastGradientArgsBroadcastGradientArgs5Optimizer/gradients/output_layer/truediv_2_grad/Shape7Optimizer/gradients/output_layer/truediv_2_grad/Shape_1*
T0
�
7Optimizer/gradients/output_layer/truediv_2_grad/RealDivRealDivHOptimizer/gradients/output_layer/truediv_3_grad/tuple/control_dependencyoutput_layer/Sum_1*
T0
�
3Optimizer/gradients/output_layer/truediv_2_grad/SumSum7Optimizer/gradients/output_layer/truediv_2_grad/RealDivEOptimizer/gradients/output_layer/truediv_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
7Optimizer/gradients/output_layer/truediv_2_grad/ReshapeReshape3Optimizer/gradients/output_layer/truediv_2_grad/Sum5Optimizer/gradients/output_layer/truediv_2_grad/Shape*
T0*
Tshape0
W
3Optimizer/gradients/output_layer/truediv_2_grad/NegNegoutput_layer/Sum_3*
T0
�
9Optimizer/gradients/output_layer/truediv_2_grad/RealDiv_1RealDiv3Optimizer/gradients/output_layer/truediv_2_grad/Negoutput_layer/Sum_1*
T0
�
9Optimizer/gradients/output_layer/truediv_2_grad/RealDiv_2RealDiv9Optimizer/gradients/output_layer/truediv_2_grad/RealDiv_1output_layer/Sum_1*
T0
�
3Optimizer/gradients/output_layer/truediv_2_grad/mulMulHOptimizer/gradients/output_layer/truediv_3_grad/tuple/control_dependency9Optimizer/gradients/output_layer/truediv_2_grad/RealDiv_2*
T0
�
5Optimizer/gradients/output_layer/truediv_2_grad/Sum_1Sum3Optimizer/gradients/output_layer/truediv_2_grad/mulGOptimizer/gradients/output_layer/truediv_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
9Optimizer/gradients/output_layer/truediv_2_grad/Reshape_1Reshape5Optimizer/gradients/output_layer/truediv_2_grad/Sum_17Optimizer/gradients/output_layer/truediv_2_grad/Shape_1*
T0*
Tshape0
�
@Optimizer/gradients/output_layer/truediv_2_grad/tuple/group_depsNoOp8^Optimizer/gradients/output_layer/truediv_2_grad/Reshape:^Optimizer/gradients/output_layer/truediv_2_grad/Reshape_1
�
HOptimizer/gradients/output_layer/truediv_2_grad/tuple/control_dependencyIdentity7Optimizer/gradients/output_layer/truediv_2_grad/ReshapeA^Optimizer/gradients/output_layer/truediv_2_grad/tuple/group_deps*
T0*J
_class@
><loc:@Optimizer/gradients/output_layer/truediv_2_grad/Reshape
�
JOptimizer/gradients/output_layer/truediv_2_grad/tuple/control_dependency_1Identity9Optimizer/gradients/output_layer/truediv_2_grad/Reshape_1A^Optimizer/gradients/output_layer/truediv_2_grad/tuple/group_deps*
T0*L
_classB
@>loc:@Optimizer/gradients/output_layer/truediv_2_grad/Reshape_1
{
9Optimizer/gradients/output_layer/logistic_loss_grad/ShapeShapeoutput_layer/logistic_loss/sub*
T0*
out_type0

;Optimizer/gradients/output_layer/logistic_loss_grad/Shape_1Shape output_layer/logistic_loss/Log1p*
T0*
out_type0
�
IOptimizer/gradients/output_layer/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs9Optimizer/gradients/output_layer/logistic_loss_grad/Shape;Optimizer/gradients/output_layer/logistic_loss_grad/Shape_1*
T0
�
7Optimizer/gradients/output_layer/logistic_loss_grad/SumSum2Optimizer/gradients/output_layer/Mean_grad/truedivIOptimizer/gradients/output_layer/logistic_loss_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
;Optimizer/gradients/output_layer/logistic_loss_grad/ReshapeReshape7Optimizer/gradients/output_layer/logistic_loss_grad/Sum9Optimizer/gradients/output_layer/logistic_loss_grad/Shape*
T0*
Tshape0
�
9Optimizer/gradients/output_layer/logistic_loss_grad/Sum_1Sum2Optimizer/gradients/output_layer/Mean_grad/truedivKOptimizer/gradients/output_layer/logistic_loss_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
=Optimizer/gradients/output_layer/logistic_loss_grad/Reshape_1Reshape9Optimizer/gradients/output_layer/logistic_loss_grad/Sum_1;Optimizer/gradients/output_layer/logistic_loss_grad/Shape_1*
T0*
Tshape0
�
DOptimizer/gradients/output_layer/logistic_loss_grad/tuple/group_depsNoOp<^Optimizer/gradients/output_layer/logistic_loss_grad/Reshape>^Optimizer/gradients/output_layer/logistic_loss_grad/Reshape_1
�
LOptimizer/gradients/output_layer/logistic_loss_grad/tuple/control_dependencyIdentity;Optimizer/gradients/output_layer/logistic_loss_grad/ReshapeE^Optimizer/gradients/output_layer/logistic_loss_grad/tuple/group_deps*
T0*N
_classD
B@loc:@Optimizer/gradients/output_layer/logistic_loss_grad/Reshape
�
NOptimizer/gradients/output_layer/logistic_loss_grad/tuple/control_dependency_1Identity=Optimizer/gradients/output_layer/logistic_loss_grad/Reshape_1E^Optimizer/gradients/output_layer/logistic_loss_grad/tuple/group_deps*
T0*P
_classF
DBloc:@Optimizer/gradients/output_layer/logistic_loss_grad/Reshape_1
h
:Optimizer/gradients/output_layer/Mean_1_grad/Reshape/shapeConst*
dtype0*
valueB:
�
4Optimizer/gradients/output_layer/Mean_1_grad/ReshapeReshapeDOptimizer/gradients/output_layer/mul_grad/tuple/control_dependency_1:Optimizer/gradients/output_layer/Mean_1_grad/Reshape/shape*
T0*
Tshape0
r
2Optimizer/gradients/output_layer/Mean_1_grad/ShapeShapeoutput_layer/logistic_loss_1*
T0*
out_type0
�
1Optimizer/gradients/output_layer/Mean_1_grad/TileTile4Optimizer/gradients/output_layer/Mean_1_grad/Reshape2Optimizer/gradients/output_layer/Mean_1_grad/Shape*
T0*

Tmultiples0
t
4Optimizer/gradients/output_layer/Mean_1_grad/Shape_1Shapeoutput_layer/logistic_loss_1*
T0*
out_type0
]
4Optimizer/gradients/output_layer/Mean_1_grad/Shape_2Const*
dtype0*
valueB 
`
2Optimizer/gradients/output_layer/Mean_1_grad/ConstConst*
dtype0*
valueB: 
�
1Optimizer/gradients/output_layer/Mean_1_grad/ProdProd4Optimizer/gradients/output_layer/Mean_1_grad/Shape_12Optimizer/gradients/output_layer/Mean_1_grad/Const*
T0*

Tidx0*
	keep_dims( 
b
4Optimizer/gradients/output_layer/Mean_1_grad/Const_1Const*
dtype0*
valueB: 
�
3Optimizer/gradients/output_layer/Mean_1_grad/Prod_1Prod4Optimizer/gradients/output_layer/Mean_1_grad/Shape_24Optimizer/gradients/output_layer/Mean_1_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
`
6Optimizer/gradients/output_layer/Mean_1_grad/Maximum/yConst*
dtype0*
value	B :
�
4Optimizer/gradients/output_layer/Mean_1_grad/MaximumMaximum3Optimizer/gradients/output_layer/Mean_1_grad/Prod_16Optimizer/gradients/output_layer/Mean_1_grad/Maximum/y*
T0
�
5Optimizer/gradients/output_layer/Mean_1_grad/floordivFloorDiv1Optimizer/gradients/output_layer/Mean_1_grad/Prod4Optimizer/gradients/output_layer/Mean_1_grad/Maximum*
T0
�
1Optimizer/gradients/output_layer/Mean_1_grad/CastCast5Optimizer/gradients/output_layer/Mean_1_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
�
4Optimizer/gradients/output_layer/Mean_1_grad/truedivRealDiv1Optimizer/gradients/output_layer/Mean_1_grad/Tile1Optimizer/gradients/output_layer/Mean_1_grad/Cast*
T0
\
3Optimizer/gradients/output_layer/truediv_grad/ShapeConst*
dtype0*
valueB 
^
5Optimizer/gradients/output_layer/truediv_grad/Shape_1Const*
dtype0*
valueB 
�
COptimizer/gradients/output_layer/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs3Optimizer/gradients/output_layer/truediv_grad/Shape5Optimizer/gradients/output_layer/truediv_grad/Shape_1*
T0
�
5Optimizer/gradients/output_layer/truediv_grad/RealDivRealDivHOptimizer/gradients/output_layer/truediv_1_grad/tuple/control_dependencyoutput_layer/Sum_1*
T0
�
1Optimizer/gradients/output_layer/truediv_grad/SumSum5Optimizer/gradients/output_layer/truediv_grad/RealDivCOptimizer/gradients/output_layer/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
5Optimizer/gradients/output_layer/truediv_grad/ReshapeReshape1Optimizer/gradients/output_layer/truediv_grad/Sum3Optimizer/gradients/output_layer/truediv_grad/Shape*
T0*
Tshape0
U
1Optimizer/gradients/output_layer/truediv_grad/NegNegoutput_layer/Sum_2*
T0
�
7Optimizer/gradients/output_layer/truediv_grad/RealDiv_1RealDiv1Optimizer/gradients/output_layer/truediv_grad/Negoutput_layer/Sum_1*
T0
�
7Optimizer/gradients/output_layer/truediv_grad/RealDiv_2RealDiv7Optimizer/gradients/output_layer/truediv_grad/RealDiv_1output_layer/Sum_1*
T0
�
1Optimizer/gradients/output_layer/truediv_grad/mulMulHOptimizer/gradients/output_layer/truediv_1_grad/tuple/control_dependency7Optimizer/gradients/output_layer/truediv_grad/RealDiv_2*
T0
�
3Optimizer/gradients/output_layer/truediv_grad/Sum_1Sum1Optimizer/gradients/output_layer/truediv_grad/mulEOptimizer/gradients/output_layer/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
7Optimizer/gradients/output_layer/truediv_grad/Reshape_1Reshape3Optimizer/gradients/output_layer/truediv_grad/Sum_15Optimizer/gradients/output_layer/truediv_grad/Shape_1*
T0*
Tshape0
�
>Optimizer/gradients/output_layer/truediv_grad/tuple/group_depsNoOp6^Optimizer/gradients/output_layer/truediv_grad/Reshape8^Optimizer/gradients/output_layer/truediv_grad/Reshape_1
�
FOptimizer/gradients/output_layer/truediv_grad/tuple/control_dependencyIdentity5Optimizer/gradients/output_layer/truediv_grad/Reshape?^Optimizer/gradients/output_layer/truediv_grad/tuple/group_deps*
T0*H
_class>
<:loc:@Optimizer/gradients/output_layer/truediv_grad/Reshape
�
HOptimizer/gradients/output_layer/truediv_grad/tuple/control_dependency_1Identity7Optimizer/gradients/output_layer/truediv_grad/Reshape_1?^Optimizer/gradients/output_layer/truediv_grad/tuple/group_deps*
T0*J
_class@
><loc:@Optimizer/gradients/output_layer/truediv_grad/Reshape_1
r
9Optimizer/gradients/output_layer/Sum_3_grad/Reshape/shapeConst*
dtype0*!
valueB"         
�
3Optimizer/gradients/output_layer/Sum_3_grad/ReshapeReshapeHOptimizer/gradients/output_layer/truediv_2_grad/tuple/control_dependency9Optimizer/gradients/output_layer/Sum_3_grad/Reshape/shape*
T0*
Tshape0
h
1Optimizer/gradients/output_layer/Sum_3_grad/ShapeShapeoutput_layer/Square*
T0*
out_type0
�
0Optimizer/gradients/output_layer/Sum_3_grad/TileTile3Optimizer/gradients/output_layer/Sum_3_grad/Reshape1Optimizer/gradients/output_layer/Sum_3_grad/Shape*
T0*

Tmultiples0
�
=Optimizer/gradients/output_layer/logistic_loss/sub_grad/ShapeShape!output_layer/logistic_loss/Select*
T0*
out_type0
�
?Optimizer/gradients/output_layer/logistic_loss/sub_grad/Shape_1Shapeoutput_layer/logistic_loss/mul*
T0*
out_type0
�
MOptimizer/gradients/output_layer/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgs=Optimizer/gradients/output_layer/logistic_loss/sub_grad/Shape?Optimizer/gradients/output_layer/logistic_loss/sub_grad/Shape_1*
T0
�
;Optimizer/gradients/output_layer/logistic_loss/sub_grad/SumSumLOptimizer/gradients/output_layer/logistic_loss_grad/tuple/control_dependencyMOptimizer/gradients/output_layer/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
?Optimizer/gradients/output_layer/logistic_loss/sub_grad/ReshapeReshape;Optimizer/gradients/output_layer/logistic_loss/sub_grad/Sum=Optimizer/gradients/output_layer/logistic_loss/sub_grad/Shape*
T0*
Tshape0
�
;Optimizer/gradients/output_layer/logistic_loss/sub_grad/NegNegLOptimizer/gradients/output_layer/logistic_loss_grad/tuple/control_dependency*
T0
�
=Optimizer/gradients/output_layer/logistic_loss/sub_grad/Sum_1Sum;Optimizer/gradients/output_layer/logistic_loss/sub_grad/NegOOptimizer/gradients/output_layer/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
AOptimizer/gradients/output_layer/logistic_loss/sub_grad/Reshape_1Reshape=Optimizer/gradients/output_layer/logistic_loss/sub_grad/Sum_1?Optimizer/gradients/output_layer/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0
�
HOptimizer/gradients/output_layer/logistic_loss/sub_grad/tuple/group_depsNoOp@^Optimizer/gradients/output_layer/logistic_loss/sub_grad/ReshapeB^Optimizer/gradients/output_layer/logistic_loss/sub_grad/Reshape_1
�
POptimizer/gradients/output_layer/logistic_loss/sub_grad/tuple/control_dependencyIdentity?Optimizer/gradients/output_layer/logistic_loss/sub_grad/ReshapeI^Optimizer/gradients/output_layer/logistic_loss/sub_grad/tuple/group_deps*
T0*R
_classH
FDloc:@Optimizer/gradients/output_layer/logistic_loss/sub_grad/Reshape
�
ROptimizer/gradients/output_layer/logistic_loss/sub_grad/tuple/control_dependency_1IdentityAOptimizer/gradients/output_layer/logistic_loss/sub_grad/Reshape_1I^Optimizer/gradients/output_layer/logistic_loss/sub_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@Optimizer/gradients/output_layer/logistic_loss/sub_grad/Reshape_1
�
?Optimizer/gradients/output_layer/logistic_loss/Log1p_grad/add/xConstO^Optimizer/gradients/output_layer/logistic_loss_grad/tuple/control_dependency_1*
dtype0*
valueB
 *  �?
�
=Optimizer/gradients/output_layer/logistic_loss/Log1p_grad/addAddV2?Optimizer/gradients/output_layer/logistic_loss/Log1p_grad/add/xoutput_layer/logistic_loss/Exp*
T0
�
DOptimizer/gradients/output_layer/logistic_loss/Log1p_grad/Reciprocal
Reciprocal=Optimizer/gradients/output_layer/logistic_loss/Log1p_grad/add*
T0
�
=Optimizer/gradients/output_layer/logistic_loss/Log1p_grad/mulMulNOptimizer/gradients/output_layer/logistic_loss_grad/tuple/control_dependency_1DOptimizer/gradients/output_layer/logistic_loss/Log1p_grad/Reciprocal*
T0

;Optimizer/gradients/output_layer/logistic_loss_1_grad/ShapeShape output_layer/logistic_loss_1/sub*
T0*
out_type0
�
=Optimizer/gradients/output_layer/logistic_loss_1_grad/Shape_1Shape"output_layer/logistic_loss_1/Log1p*
T0*
out_type0
�
KOptimizer/gradients/output_layer/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs;Optimizer/gradients/output_layer/logistic_loss_1_grad/Shape=Optimizer/gradients/output_layer/logistic_loss_1_grad/Shape_1*
T0
�
9Optimizer/gradients/output_layer/logistic_loss_1_grad/SumSum4Optimizer/gradients/output_layer/Mean_1_grad/truedivKOptimizer/gradients/output_layer/logistic_loss_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
=Optimizer/gradients/output_layer/logistic_loss_1_grad/ReshapeReshape9Optimizer/gradients/output_layer/logistic_loss_1_grad/Sum;Optimizer/gradients/output_layer/logistic_loss_1_grad/Shape*
T0*
Tshape0
�
;Optimizer/gradients/output_layer/logistic_loss_1_grad/Sum_1Sum4Optimizer/gradients/output_layer/Mean_1_grad/truedivMOptimizer/gradients/output_layer/logistic_loss_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
?Optimizer/gradients/output_layer/logistic_loss_1_grad/Reshape_1Reshape;Optimizer/gradients/output_layer/logistic_loss_1_grad/Sum_1=Optimizer/gradients/output_layer/logistic_loss_1_grad/Shape_1*
T0*
Tshape0
�
FOptimizer/gradients/output_layer/logistic_loss_1_grad/tuple/group_depsNoOp>^Optimizer/gradients/output_layer/logistic_loss_1_grad/Reshape@^Optimizer/gradients/output_layer/logistic_loss_1_grad/Reshape_1
�
NOptimizer/gradients/output_layer/logistic_loss_1_grad/tuple/control_dependencyIdentity=Optimizer/gradients/output_layer/logistic_loss_1_grad/ReshapeG^Optimizer/gradients/output_layer/logistic_loss_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@Optimizer/gradients/output_layer/logistic_loss_1_grad/Reshape
�
POptimizer/gradients/output_layer/logistic_loss_1_grad/tuple/control_dependency_1Identity?Optimizer/gradients/output_layer/logistic_loss_1_grad/Reshape_1G^Optimizer/gradients/output_layer/logistic_loss_1_grad/tuple/group_deps*
T0*R
_classH
FDloc:@Optimizer/gradients/output_layer/logistic_loss_1_grad/Reshape_1
r
9Optimizer/gradients/output_layer/Sum_2_grad/Reshape/shapeConst*
dtype0*!
valueB"         
�
3Optimizer/gradients/output_layer/Sum_2_grad/ReshapeReshapeFOptimizer/gradients/output_layer/truediv_grad/tuple/control_dependency9Optimizer/gradients/output_layer/Sum_2_grad/Reshape/shape*
T0*
Tshape0
g
1Optimizer/gradients/output_layer/Sum_2_grad/ShapeShapeoutput_layer/Abs_1*
T0*
out_type0
�
0Optimizer/gradients/output_layer/Sum_2_grad/TileTile3Optimizer/gradients/output_layer/Sum_2_grad/Reshape1Optimizer/gradients/output_layer/Sum_2_grad/Shape*
T0*

Tmultiples0
�
2Optimizer/gradients/output_layer/Square_grad/ConstConst1^Optimizer/gradients/output_layer/Sum_3_grad/Tile*
dtype0*
valueB
 *   @
�
0Optimizer/gradients/output_layer/Square_grad/MulMuloutput_layer/sub_12Optimizer/gradients/output_layer/Square_grad/Const*
T0
�
2Optimizer/gradients/output_layer/Square_grad/Mul_1Mul0Optimizer/gradients/output_layer/Sum_3_grad/Tile0Optimizer/gradients/output_layer/Square_grad/Mul*
T0
r
EOptimizer/gradients/output_layer/logistic_loss/Select_grad/zeros_like	ZerosLikeoutput_layer/GatherNd*
T0
�
AOptimizer/gradients/output_layer/logistic_loss/Select_grad/SelectSelect'output_layer/logistic_loss/GreaterEqualPOptimizer/gradients/output_layer/logistic_loss/sub_grad/tuple/control_dependencyEOptimizer/gradients/output_layer/logistic_loss/Select_grad/zeros_like*
T0
�
COptimizer/gradients/output_layer/logistic_loss/Select_grad/Select_1Select'output_layer/logistic_loss/GreaterEqualEOptimizer/gradients/output_layer/logistic_loss/Select_grad/zeros_likePOptimizer/gradients/output_layer/logistic_loss/sub_grad/tuple/control_dependency*
T0
�
KOptimizer/gradients/output_layer/logistic_loss/Select_grad/tuple/group_depsNoOpB^Optimizer/gradients/output_layer/logistic_loss/Select_grad/SelectD^Optimizer/gradients/output_layer/logistic_loss/Select_grad/Select_1
�
SOptimizer/gradients/output_layer/logistic_loss/Select_grad/tuple/control_dependencyIdentityAOptimizer/gradients/output_layer/logistic_loss/Select_grad/SelectL^Optimizer/gradients/output_layer/logistic_loss/Select_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@Optimizer/gradients/output_layer/logistic_loss/Select_grad/Select
�
UOptimizer/gradients/output_layer/logistic_loss/Select_grad/tuple/control_dependency_1IdentityCOptimizer/gradients/output_layer/logistic_loss/Select_grad/Select_1L^Optimizer/gradients/output_layer/logistic_loss/Select_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/output_layer/logistic_loss/Select_grad/Select_1
v
=Optimizer/gradients/output_layer/logistic_loss/mul_grad/ShapeShapeoutput_layer/GatherNd*
T0*
out_type0
z
?Optimizer/gradients/output_layer/logistic_loss/mul_grad/Shape_1Shapeoutput_layer/GatherNd_2*
T0*
out_type0
�
MOptimizer/gradients/output_layer/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgs=Optimizer/gradients/output_layer/logistic_loss/mul_grad/Shape?Optimizer/gradients/output_layer/logistic_loss/mul_grad/Shape_1*
T0
�
;Optimizer/gradients/output_layer/logistic_loss/mul_grad/MulMulROptimizer/gradients/output_layer/logistic_loss/sub_grad/tuple/control_dependency_1output_layer/GatherNd_2*
T0
�
;Optimizer/gradients/output_layer/logistic_loss/mul_grad/SumSum;Optimizer/gradients/output_layer/logistic_loss/mul_grad/MulMOptimizer/gradients/output_layer/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
?Optimizer/gradients/output_layer/logistic_loss/mul_grad/ReshapeReshape;Optimizer/gradients/output_layer/logistic_loss/mul_grad/Sum=Optimizer/gradients/output_layer/logistic_loss/mul_grad/Shape*
T0*
Tshape0
�
=Optimizer/gradients/output_layer/logistic_loss/mul_grad/Mul_1Muloutput_layer/GatherNdROptimizer/gradients/output_layer/logistic_loss/sub_grad/tuple/control_dependency_1*
T0
�
=Optimizer/gradients/output_layer/logistic_loss/mul_grad/Sum_1Sum=Optimizer/gradients/output_layer/logistic_loss/mul_grad/Mul_1OOptimizer/gradients/output_layer/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
AOptimizer/gradients/output_layer/logistic_loss/mul_grad/Reshape_1Reshape=Optimizer/gradients/output_layer/logistic_loss/mul_grad/Sum_1?Optimizer/gradients/output_layer/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
�
HOptimizer/gradients/output_layer/logistic_loss/mul_grad/tuple/group_depsNoOp@^Optimizer/gradients/output_layer/logistic_loss/mul_grad/ReshapeB^Optimizer/gradients/output_layer/logistic_loss/mul_grad/Reshape_1
�
POptimizer/gradients/output_layer/logistic_loss/mul_grad/tuple/control_dependencyIdentity?Optimizer/gradients/output_layer/logistic_loss/mul_grad/ReshapeI^Optimizer/gradients/output_layer/logistic_loss/mul_grad/tuple/group_deps*
T0*R
_classH
FDloc:@Optimizer/gradients/output_layer/logistic_loss/mul_grad/Reshape
�
ROptimizer/gradients/output_layer/logistic_loss/mul_grad/tuple/control_dependency_1IdentityAOptimizer/gradients/output_layer/logistic_loss/mul_grad/Reshape_1I^Optimizer/gradients/output_layer/logistic_loss/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@Optimizer/gradients/output_layer/logistic_loss/mul_grad/Reshape_1
�
;Optimizer/gradients/output_layer/logistic_loss/Exp_grad/mulMul=Optimizer/gradients/output_layer/logistic_loss/Log1p_grad/muloutput_layer/logistic_loss/Exp*
T0
�
?Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/ShapeShape#output_layer/logistic_loss_1/Select*
T0*
out_type0
�
AOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/Shape_1Shape output_layer/logistic_loss_1/mul*
T0*
out_type0
�
OOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs?Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/ShapeAOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/Shape_1*
T0
�
=Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/SumSumNOptimizer/gradients/output_layer/logistic_loss_1_grad/tuple/control_dependencyOOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
AOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/ReshapeReshape=Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/Sum?Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/Shape*
T0*
Tshape0
�
=Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/NegNegNOptimizer/gradients/output_layer/logistic_loss_1_grad/tuple/control_dependency*
T0
�
?Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/Sum_1Sum=Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/NegQOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
COptimizer/gradients/output_layer/logistic_loss_1/sub_grad/Reshape_1Reshape?Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/Sum_1AOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/Shape_1*
T0*
Tshape0
�
JOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/tuple/group_depsNoOpB^Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/ReshapeD^Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/Reshape_1
�
ROptimizer/gradients/output_layer/logistic_loss_1/sub_grad/tuple/control_dependencyIdentityAOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/ReshapeK^Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/Reshape
�
TOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/tuple/control_dependency_1IdentityCOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/Reshape_1K^Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/output_layer/logistic_loss_1/sub_grad/Reshape_1
�
AOptimizer/gradients/output_layer/logistic_loss_1/Log1p_grad/add/xConstQ^Optimizer/gradients/output_layer/logistic_loss_1_grad/tuple/control_dependency_1*
dtype0*
valueB
 *  �?
�
?Optimizer/gradients/output_layer/logistic_loss_1/Log1p_grad/addAddV2AOptimizer/gradients/output_layer/logistic_loss_1/Log1p_grad/add/x output_layer/logistic_loss_1/Exp*
T0
�
FOptimizer/gradients/output_layer/logistic_loss_1/Log1p_grad/Reciprocal
Reciprocal?Optimizer/gradients/output_layer/logistic_loss_1/Log1p_grad/add*
T0
�
?Optimizer/gradients/output_layer/logistic_loss_1/Log1p_grad/mulMulPOptimizer/gradients/output_layer/logistic_loss_1_grad/tuple/control_dependency_1FOptimizer/gradients/output_layer/logistic_loss_1/Log1p_grad/Reciprocal*
T0
S
0Optimizer/gradients/output_layer/Abs_1_grad/SignSignoutput_layer/sub*
T0
�
/Optimizer/gradients/output_layer/Abs_1_grad/mulMul0Optimizer/gradients/output_layer/Sum_2_grad/Tile0Optimizer/gradients/output_layer/Abs_1_grad/Sign*
T0
q
1Optimizer/gradients/output_layer/sub_1_grad/ShapeShapeoutput_layer/strided_slice_5*
T0*
out_type0
s
3Optimizer/gradients/output_layer/sub_1_grad/Shape_1Shapeoutput_layer/strided_slice_6*
T0*
out_type0
�
AOptimizer/gradients/output_layer/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs1Optimizer/gradients/output_layer/sub_1_grad/Shape3Optimizer/gradients/output_layer/sub_1_grad/Shape_1*
T0
�
/Optimizer/gradients/output_layer/sub_1_grad/SumSum2Optimizer/gradients/output_layer/Square_grad/Mul_1AOptimizer/gradients/output_layer/sub_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
3Optimizer/gradients/output_layer/sub_1_grad/ReshapeReshape/Optimizer/gradients/output_layer/sub_1_grad/Sum1Optimizer/gradients/output_layer/sub_1_grad/Shape*
T0*
Tshape0
s
/Optimizer/gradients/output_layer/sub_1_grad/NegNeg2Optimizer/gradients/output_layer/Square_grad/Mul_1*
T0
�
1Optimizer/gradients/output_layer/sub_1_grad/Sum_1Sum/Optimizer/gradients/output_layer/sub_1_grad/NegCOptimizer/gradients/output_layer/sub_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
5Optimizer/gradients/output_layer/sub_1_grad/Reshape_1Reshape1Optimizer/gradients/output_layer/sub_1_grad/Sum_13Optimizer/gradients/output_layer/sub_1_grad/Shape_1*
T0*
Tshape0
�
<Optimizer/gradients/output_layer/sub_1_grad/tuple/group_depsNoOp4^Optimizer/gradients/output_layer/sub_1_grad/Reshape6^Optimizer/gradients/output_layer/sub_1_grad/Reshape_1
�
DOptimizer/gradients/output_layer/sub_1_grad/tuple/control_dependencyIdentity3Optimizer/gradients/output_layer/sub_1_grad/Reshape=^Optimizer/gradients/output_layer/sub_1_grad/tuple/group_deps*
T0*F
_class<
:8loc:@Optimizer/gradients/output_layer/sub_1_grad/Reshape
�
FOptimizer/gradients/output_layer/sub_1_grad/tuple/control_dependency_1Identity5Optimizer/gradients/output_layer/sub_1_grad/Reshape_1=^Optimizer/gradients/output_layer/sub_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@Optimizer/gradients/output_layer/sub_1_grad/Reshape_1
}
GOptimizer/gradients/output_layer/logistic_loss/Select_1_grad/zeros_like	ZerosLikeoutput_layer/logistic_loss/Neg*
T0
�
COptimizer/gradients/output_layer/logistic_loss/Select_1_grad/SelectSelect'output_layer/logistic_loss/GreaterEqual;Optimizer/gradients/output_layer/logistic_loss/Exp_grad/mulGOptimizer/gradients/output_layer/logistic_loss/Select_1_grad/zeros_like*
T0
�
EOptimizer/gradients/output_layer/logistic_loss/Select_1_grad/Select_1Select'output_layer/logistic_loss/GreaterEqualGOptimizer/gradients/output_layer/logistic_loss/Select_1_grad/zeros_like;Optimizer/gradients/output_layer/logistic_loss/Exp_grad/mul*
T0
�
MOptimizer/gradients/output_layer/logistic_loss/Select_1_grad/tuple/group_depsNoOpD^Optimizer/gradients/output_layer/logistic_loss/Select_1_grad/SelectF^Optimizer/gradients/output_layer/logistic_loss/Select_1_grad/Select_1
�
UOptimizer/gradients/output_layer/logistic_loss/Select_1_grad/tuple/control_dependencyIdentityCOptimizer/gradients/output_layer/logistic_loss/Select_1_grad/SelectN^Optimizer/gradients/output_layer/logistic_loss/Select_1_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/output_layer/logistic_loss/Select_1_grad/Select
�
WOptimizer/gradients/output_layer/logistic_loss/Select_1_grad/tuple/control_dependency_1IdentityEOptimizer/gradients/output_layer/logistic_loss/Select_1_grad/Select_1N^Optimizer/gradients/output_layer/logistic_loss/Select_1_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer/gradients/output_layer/logistic_loss/Select_1_grad/Select_1
v
GOptimizer/gradients/output_layer/logistic_loss_1/Select_grad/zeros_like	ZerosLikeoutput_layer/GatherNd_3*
T0
�
COptimizer/gradients/output_layer/logistic_loss_1/Select_grad/SelectSelect)output_layer/logistic_loss_1/GreaterEqualROptimizer/gradients/output_layer/logistic_loss_1/sub_grad/tuple/control_dependencyGOptimizer/gradients/output_layer/logistic_loss_1/Select_grad/zeros_like*
T0
�
EOptimizer/gradients/output_layer/logistic_loss_1/Select_grad/Select_1Select)output_layer/logistic_loss_1/GreaterEqualGOptimizer/gradients/output_layer/logistic_loss_1/Select_grad/zeros_likeROptimizer/gradients/output_layer/logistic_loss_1/sub_grad/tuple/control_dependency*
T0
�
MOptimizer/gradients/output_layer/logistic_loss_1/Select_grad/tuple/group_depsNoOpD^Optimizer/gradients/output_layer/logistic_loss_1/Select_grad/SelectF^Optimizer/gradients/output_layer/logistic_loss_1/Select_grad/Select_1
�
UOptimizer/gradients/output_layer/logistic_loss_1/Select_grad/tuple/control_dependencyIdentityCOptimizer/gradients/output_layer/logistic_loss_1/Select_grad/SelectN^Optimizer/gradients/output_layer/logistic_loss_1/Select_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/output_layer/logistic_loss_1/Select_grad/Select
�
WOptimizer/gradients/output_layer/logistic_loss_1/Select_grad/tuple/control_dependency_1IdentityEOptimizer/gradients/output_layer/logistic_loss_1/Select_grad/Select_1N^Optimizer/gradients/output_layer/logistic_loss_1/Select_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer/gradients/output_layer/logistic_loss_1/Select_grad/Select_1
z
?Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/ShapeShapeoutput_layer/GatherNd_3*
T0*
out_type0
|
AOptimizer/gradients/output_layer/logistic_loss_1/mul_grad/Shape_1Shapeoutput_layer/GatherNd_5*
T0*
out_type0
�
OOptimizer/gradients/output_layer/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgs?Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/ShapeAOptimizer/gradients/output_layer/logistic_loss_1/mul_grad/Shape_1*
T0
�
=Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/MulMulTOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/tuple/control_dependency_1output_layer/GatherNd_5*
T0
�
=Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/SumSum=Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/MulOOptimizer/gradients/output_layer/logistic_loss_1/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
AOptimizer/gradients/output_layer/logistic_loss_1/mul_grad/ReshapeReshape=Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/Sum?Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/Shape*
T0*
Tshape0
�
?Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/Mul_1Muloutput_layer/GatherNd_3TOptimizer/gradients/output_layer/logistic_loss_1/sub_grad/tuple/control_dependency_1*
T0
�
?Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/Sum_1Sum?Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/Mul_1QOptimizer/gradients/output_layer/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
COptimizer/gradients/output_layer/logistic_loss_1/mul_grad/Reshape_1Reshape?Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/Sum_1AOptimizer/gradients/output_layer/logistic_loss_1/mul_grad/Shape_1*
T0*
Tshape0
�
JOptimizer/gradients/output_layer/logistic_loss_1/mul_grad/tuple/group_depsNoOpB^Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/ReshapeD^Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/Reshape_1
�
ROptimizer/gradients/output_layer/logistic_loss_1/mul_grad/tuple/control_dependencyIdentityAOptimizer/gradients/output_layer/logistic_loss_1/mul_grad/ReshapeK^Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/Reshape
�
TOptimizer/gradients/output_layer/logistic_loss_1/mul_grad/tuple/control_dependency_1IdentityCOptimizer/gradients/output_layer/logistic_loss_1/mul_grad/Reshape_1K^Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/output_layer/logistic_loss_1/mul_grad/Reshape_1
�
=Optimizer/gradients/output_layer/logistic_loss_1/Exp_grad/mulMul?Optimizer/gradients/output_layer/logistic_loss_1/Log1p_grad/mul output_layer/logistic_loss_1/Exp*
T0
o
/Optimizer/gradients/output_layer/sub_grad/ShapeShapeoutput_layer/strided_slice_3*
T0*
out_type0
q
1Optimizer/gradients/output_layer/sub_grad/Shape_1Shapeoutput_layer/strided_slice_4*
T0*
out_type0
�
?Optimizer/gradients/output_layer/sub_grad/BroadcastGradientArgsBroadcastGradientArgs/Optimizer/gradients/output_layer/sub_grad/Shape1Optimizer/gradients/output_layer/sub_grad/Shape_1*
T0
�
-Optimizer/gradients/output_layer/sub_grad/SumSum/Optimizer/gradients/output_layer/Abs_1_grad/mul?Optimizer/gradients/output_layer/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
1Optimizer/gradients/output_layer/sub_grad/ReshapeReshape-Optimizer/gradients/output_layer/sub_grad/Sum/Optimizer/gradients/output_layer/sub_grad/Shape*
T0*
Tshape0
n
-Optimizer/gradients/output_layer/sub_grad/NegNeg/Optimizer/gradients/output_layer/Abs_1_grad/mul*
T0
�
/Optimizer/gradients/output_layer/sub_grad/Sum_1Sum-Optimizer/gradients/output_layer/sub_grad/NegAOptimizer/gradients/output_layer/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
3Optimizer/gradients/output_layer/sub_grad/Reshape_1Reshape/Optimizer/gradients/output_layer/sub_grad/Sum_11Optimizer/gradients/output_layer/sub_grad/Shape_1*
T0*
Tshape0
�
:Optimizer/gradients/output_layer/sub_grad/tuple/group_depsNoOp2^Optimizer/gradients/output_layer/sub_grad/Reshape4^Optimizer/gradients/output_layer/sub_grad/Reshape_1
�
BOptimizer/gradients/output_layer/sub_grad/tuple/control_dependencyIdentity1Optimizer/gradients/output_layer/sub_grad/Reshape;^Optimizer/gradients/output_layer/sub_grad/tuple/group_deps*
T0*D
_class:
86loc:@Optimizer/gradients/output_layer/sub_grad/Reshape
�
DOptimizer/gradients/output_layer/sub_grad/tuple/control_dependency_1Identity3Optimizer/gradients/output_layer/sub_grad/Reshape_1;^Optimizer/gradients/output_layer/sub_grad/tuple/group_deps*
T0*F
_class<
:8loc:@Optimizer/gradients/output_layer/sub_grad/Reshape_1
q
;Optimizer/gradients/output_layer/strided_slice_5_grad/ShapeShapeoutput_layer/preds*
T0*
out_type0
�
LOptimizer/gradients/output_layer/strided_slice_5_grad/StridedSliceGrad/beginConst*
dtype0*!
valueB"           
�
JOptimizer/gradients/output_layer/strided_slice_5_grad/StridedSliceGrad/endConst*
dtype0*!
valueB"            
�
NOptimizer/gradients/output_layer/strided_slice_5_grad/StridedSliceGrad/stridesConst*
dtype0*!
valueB"         
�
FOptimizer/gradients/output_layer/strided_slice_5_grad/StridedSliceGradStridedSliceGrad;Optimizer/gradients/output_layer/strided_slice_5_grad/ShapeLOptimizer/gradients/output_layer/strided_slice_5_grad/StridedSliceGrad/beginJOptimizer/gradients/output_layer/strided_slice_5_grad/StridedSliceGrad/endNOptimizer/gradients/output_layer/strided_slice_5_grad/StridedSliceGrad/stridesDOptimizer/gradients/output_layer/sub_1_grad/tuple/control_dependency*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
q
;Optimizer/gradients/output_layer/strided_slice_6_grad/ShapeShapeoutput_layer/preds*
T0*
out_type0
�
LOptimizer/gradients/output_layer/strided_slice_6_grad/StridedSliceGrad/beginConst*
dtype0*!
valueB"            
�
JOptimizer/gradients/output_layer/strided_slice_6_grad/StridedSliceGrad/endConst*
dtype0*!
valueB"    ����    
�
NOptimizer/gradients/output_layer/strided_slice_6_grad/StridedSliceGrad/stridesConst*
dtype0*!
valueB"         
�
FOptimizer/gradients/output_layer/strided_slice_6_grad/StridedSliceGradStridedSliceGrad;Optimizer/gradients/output_layer/strided_slice_6_grad/ShapeLOptimizer/gradients/output_layer/strided_slice_6_grad/StridedSliceGrad/beginJOptimizer/gradients/output_layer/strided_slice_6_grad/StridedSliceGrad/endNOptimizer/gradients/output_layer/strided_slice_6_grad/StridedSliceGrad/stridesFOptimizer/gradients/output_layer/sub_1_grad/tuple/control_dependency_1*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
�
;Optimizer/gradients/output_layer/logistic_loss/Neg_grad/NegNegUOptimizer/gradients/output_layer/logistic_loss/Select_1_grad/tuple/control_dependency*
T0
�
IOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/zeros_like	ZerosLike output_layer/logistic_loss_1/Neg*
T0
�
EOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/SelectSelect)output_layer/logistic_loss_1/GreaterEqual=Optimizer/gradients/output_layer/logistic_loss_1/Exp_grad/mulIOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/zeros_like*
T0
�
GOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/Select_1Select)output_layer/logistic_loss_1/GreaterEqualIOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/zeros_like=Optimizer/gradients/output_layer/logistic_loss_1/Exp_grad/mul*
T0
�
OOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/tuple/group_depsNoOpF^Optimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/SelectH^Optimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/Select_1
�
WOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentityEOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/SelectP^Optimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/Select
�
YOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/tuple/control_dependency_1IdentityGOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/Select_1P^Optimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/Select_1
q
;Optimizer/gradients/output_layer/strided_slice_3_grad/ShapeShapeoutput_layer/preds*
T0*
out_type0
�
LOptimizer/gradients/output_layer/strided_slice_3_grad/StridedSliceGrad/beginConst*
dtype0*!
valueB"           
�
JOptimizer/gradients/output_layer/strided_slice_3_grad/StridedSliceGrad/endConst*
dtype0*!
valueB"            
�
NOptimizer/gradients/output_layer/strided_slice_3_grad/StridedSliceGrad/stridesConst*
dtype0*!
valueB"         
�
FOptimizer/gradients/output_layer/strided_slice_3_grad/StridedSliceGradStridedSliceGrad;Optimizer/gradients/output_layer/strided_slice_3_grad/ShapeLOptimizer/gradients/output_layer/strided_slice_3_grad/StridedSliceGrad/beginJOptimizer/gradients/output_layer/strided_slice_3_grad/StridedSliceGrad/endNOptimizer/gradients/output_layer/strided_slice_3_grad/StridedSliceGrad/stridesBOptimizer/gradients/output_layer/sub_grad/tuple/control_dependency*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
q
;Optimizer/gradients/output_layer/strided_slice_4_grad/ShapeShapeoutput_layer/preds*
T0*
out_type0
�
LOptimizer/gradients/output_layer/strided_slice_4_grad/StridedSliceGrad/beginConst*
dtype0*!
valueB"            
�
JOptimizer/gradients/output_layer/strided_slice_4_grad/StridedSliceGrad/endConst*
dtype0*!
valueB"    ����    
�
NOptimizer/gradients/output_layer/strided_slice_4_grad/StridedSliceGrad/stridesConst*
dtype0*!
valueB"         
�
FOptimizer/gradients/output_layer/strided_slice_4_grad/StridedSliceGradStridedSliceGrad;Optimizer/gradients/output_layer/strided_slice_4_grad/ShapeLOptimizer/gradients/output_layer/strided_slice_4_grad/StridedSliceGrad/beginJOptimizer/gradients/output_layer/strided_slice_4_grad/StridedSliceGrad/endNOptimizer/gradients/output_layer/strided_slice_4_grad/StridedSliceGrad/stridesDOptimizer/gradients/output_layer/sub_grad/tuple/control_dependency_1*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
�
=Optimizer/gradients/output_layer/logistic_loss_1/Neg_grad/NegNegWOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/tuple/control_dependency*
T0
�
Optimizer/gradients/AddNAddNFOptimizer/gradients/output_layer/strided_slice_5_grad/StridedSliceGradFOptimizer/gradients/output_layer/strided_slice_6_grad/StridedSliceGradFOptimizer/gradients/output_layer/strided_slice_3_grad/StridedSliceGradFOptimizer/gradients/output_layer/strided_slice_4_grad/StridedSliceGrad*
N*
T0*Y
_classO
MKloc:@Optimizer/gradients/output_layer/strided_slice_5_grad/StridedSliceGrad
}
7Optimizer/gradients/output_layer/preds_grad/SigmoidGradSigmoidGradoutput_layer/predsOptimizer/gradients/AddN*
T0
�
Optimizer/gradients/AddN_1AddNSOptimizer/gradients/output_layer/logistic_loss/Select_grad/tuple/control_dependencyPOptimizer/gradients/output_layer/logistic_loss/mul_grad/tuple/control_dependencyWOptimizer/gradients/output_layer/logistic_loss/Select_1_grad/tuple/control_dependency_1;Optimizer/gradients/output_layer/logistic_loss/Neg_grad/Neg*
N*
T0*T
_classJ
HFloc:@Optimizer/gradients/output_layer/logistic_loss/Select_grad/Select
n
4Optimizer/gradients/output_layer/GatherNd_grad/ShapeShapeoutput_layer/Reshape_1*
T0*
out_type0	
�
8Optimizer/gradients/output_layer/GatherNd_grad/ScatterNd	ScatterNdoutput_layer/WhereOptimizer/gradients/AddN_14Optimizer/gradients/output_layer/GatherNd_grad/Shape*
T0*
Tindices0	
�
Optimizer/gradients/AddN_2AddNUOptimizer/gradients/output_layer/logistic_loss_1/Select_grad/tuple/control_dependencyROptimizer/gradients/output_layer/logistic_loss_1/mul_grad/tuple/control_dependencyYOptimizer/gradients/output_layer/logistic_loss_1/Select_1_grad/tuple/control_dependency_1=Optimizer/gradients/output_layer/logistic_loss_1/Neg_grad/Neg*
N*
T0*V
_classL
JHloc:@Optimizer/gradients/output_layer/logistic_loss_1/Select_grad/Select
p
6Optimizer/gradients/output_layer/GatherNd_3_grad/ShapeShapeoutput_layer/Reshape_1*
T0*
out_type0	
�
:Optimizer/gradients/output_layer/GatherNd_3_grad/ScatterNd	ScatterNdoutput_layer/Where_1Optimizer/gradients/AddN_26Optimizer/gradients/output_layer/GatherNd_3_grad/Shape*
T0*
Tindices0	
�
Optimizer/gradients/AddN_3AddN7Optimizer/gradients/output_layer/preds_grad/SigmoidGrad8Optimizer/gradients/output_layer/GatherNd_grad/ScatterNd:Optimizer/gradients/output_layer/GatherNd_3_grad/ScatterNd*
N*
T0*J
_class@
><loc:@Optimizer/gradients/output_layer/preds_grad/SigmoidGrad
i
5Optimizer/gradients/output_layer/Reshape_1_grad/ShapeShapeoutput_layer/add*
T0*
out_type0
�
7Optimizer/gradients/output_layer/Reshape_1_grad/ReshapeReshapeOptimizer/gradients/AddN_35Optimizer/gradients/output_layer/Reshape_1_grad/Shape*
T0*
Tshape0
f
/Optimizer/gradients/output_layer/add_grad/ShapeShapeoutput_layer/MatMul*
T0*
out_type0
m
1Optimizer/gradients/output_layer/add_grad/Shape_1Shapeoutput_layer/biases/read*
T0*
out_type0
�
?Optimizer/gradients/output_layer/add_grad/BroadcastGradientArgsBroadcastGradientArgs/Optimizer/gradients/output_layer/add_grad/Shape1Optimizer/gradients/output_layer/add_grad/Shape_1*
T0
�
-Optimizer/gradients/output_layer/add_grad/SumSum7Optimizer/gradients/output_layer/Reshape_1_grad/Reshape?Optimizer/gradients/output_layer/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
1Optimizer/gradients/output_layer/add_grad/ReshapeReshape-Optimizer/gradients/output_layer/add_grad/Sum/Optimizer/gradients/output_layer/add_grad/Shape*
T0*
Tshape0
�
/Optimizer/gradients/output_layer/add_grad/Sum_1Sum7Optimizer/gradients/output_layer/Reshape_1_grad/ReshapeAOptimizer/gradients/output_layer/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
3Optimizer/gradients/output_layer/add_grad/Reshape_1Reshape/Optimizer/gradients/output_layer/add_grad/Sum_11Optimizer/gradients/output_layer/add_grad/Shape_1*
T0*
Tshape0
�
:Optimizer/gradients/output_layer/add_grad/tuple/group_depsNoOp2^Optimizer/gradients/output_layer/add_grad/Reshape4^Optimizer/gradients/output_layer/add_grad/Reshape_1
�
BOptimizer/gradients/output_layer/add_grad/tuple/control_dependencyIdentity1Optimizer/gradients/output_layer/add_grad/Reshape;^Optimizer/gradients/output_layer/add_grad/tuple/group_deps*
T0*D
_class:
86loc:@Optimizer/gradients/output_layer/add_grad/Reshape
�
DOptimizer/gradients/output_layer/add_grad/tuple/control_dependency_1Identity3Optimizer/gradients/output_layer/add_grad/Reshape_1;^Optimizer/gradients/output_layer/add_grad/tuple/group_deps*
T0*F
_class<
:8loc:@Optimizer/gradients/output_layer/add_grad/Reshape_1
�
3Optimizer/gradients/output_layer/MatMul_grad/MatMulMatMulBOptimizer/gradients/output_layer/add_grad/tuple/control_dependencyoutput_layer/weights/read*
T0*
transpose_a( *
transpose_b(
�
5Optimizer/gradients/output_layer/MatMul_grad/MatMul_1MatMuloutput_layer/ReshapeBOptimizer/gradients/output_layer/add_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
=Optimizer/gradients/output_layer/MatMul_grad/tuple/group_depsNoOp4^Optimizer/gradients/output_layer/MatMul_grad/MatMul6^Optimizer/gradients/output_layer/MatMul_grad/MatMul_1
�
EOptimizer/gradients/output_layer/MatMul_grad/tuple/control_dependencyIdentity3Optimizer/gradients/output_layer/MatMul_grad/MatMul>^Optimizer/gradients/output_layer/MatMul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@Optimizer/gradients/output_layer/MatMul_grad/MatMul
�
GOptimizer/gradients/output_layer/MatMul_grad/tuple/control_dependency_1Identity5Optimizer/gradients/output_layer/MatMul_grad/MatMul_1>^Optimizer/gradients/output_layer/MatMul_grad/tuple/group_deps*
T0*H
_class>
<:loc:@Optimizer/gradients/output_layer/MatMul_grad/MatMul_1
u
3Optimizer/gradients/output_layer/Reshape_grad/ShapeShapehidden_layer_0/rnn/transpose_1*
T0*
out_type0
�
5Optimizer/gradients/output_layer/Reshape_grad/ReshapeReshapeEOptimizer/gradients/output_layer/MatMul_grad/tuple/control_dependency3Optimizer/gradients/output_layer/Reshape_grad/Shape*
T0*
Tshape0
�
IOptimizer/gradients/hidden_layer_0/rnn/transpose_1_grad/InvertPermutationInvertPermutationhidden_layer_0/rnn/concat_2*
T0
�
AOptimizer/gradients/hidden_layer_0/rnn/transpose_1_grad/transpose	Transpose5Optimizer/gradients/output_layer/Reshape_grad/ReshapeIOptimizer/gradients/hidden_layer_0/rnn/transpose_1_grad/InvertPermutation*
T0*
Tperm0
�
rOptimizer/gradients/hidden_layer_0/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3hidden_layer_0/rnn/TensorArrayhidden_layer_0/rnn/while/Exit_2*1
_class'
%#loc:@hidden_layer_0/rnn/TensorArray*
sourceOptimizer/gradients
�
nOptimizer/gradients/hidden_layer_0/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentityhidden_layer_0/rnn/while/Exit_2s^Optimizer/gradients/hidden_layer_0/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*1
_class'
%#loc:@hidden_layer_0/rnn/TensorArray
�
xOptimizer/gradients/hidden_layer_0/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3rOptimizer/gradients/hidden_layer_0/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3)hidden_layer_0/rnn/TensorArrayStack/rangeAOptimizer/gradients/hidden_layer_0/rnn/transpose_1_grad/transposenOptimizer/gradients/hidden_layer_0/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0
U
Optimizer/gradients/zeros_like	ZerosLikehidden_layer_0/rnn/while/Exit_3*
T0
W
 Optimizer/gradients/zeros_like_1	ZerosLikehidden_layer_0/rnn/while/Exit_4*
T0
�
?Optimizer/gradients/hidden_layer_0/rnn/while/Exit_2_grad/b_exitEnterxOptimizer/gradients/hidden_layer_0/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
?Optimizer/gradients/hidden_layer_0/rnn/while/Exit_3_grad/b_exitEnterOptimizer/gradients/zeros_like*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
?Optimizer/gradients/hidden_layer_0/rnn/while/Exit_4_grad/b_exitEnter Optimizer/gradients/zeros_like_1*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
COptimizer/gradients/hidden_layer_0/rnn/while/Switch_2_grad/b_switchMerge?Optimizer/gradients/hidden_layer_0/rnn/while/Exit_2_grad/b_exitJOptimizer/gradients/hidden_layer_0/rnn/while/Switch_2_grad_1/NextIteration*
N*
T0
�
COptimizer/gradients/hidden_layer_0/rnn/while/Switch_3_grad/b_switchMerge?Optimizer/gradients/hidden_layer_0/rnn/while/Exit_3_grad/b_exitJOptimizer/gradients/hidden_layer_0/rnn/while/Switch_3_grad_1/NextIteration*
N*
T0
�
COptimizer/gradients/hidden_layer_0/rnn/while/Switch_4_grad/b_switchMerge?Optimizer/gradients/hidden_layer_0/rnn/while/Exit_4_grad/b_exitJOptimizer/gradients/hidden_layer_0/rnn/while/Switch_4_grad_1/NextIteration*
N*
T0
�
@Optimizer/gradients/hidden_layer_0/rnn/while/Merge_2_grad/SwitchSwitchCOptimizer/gradients/hidden_layer_0/rnn/while/Switch_2_grad/b_switchOptimizer/gradients/b_count_2*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Switch_2_grad/b_switch
�
JOptimizer/gradients/hidden_layer_0/rnn/while/Merge_2_grad/tuple/group_depsNoOpA^Optimizer/gradients/hidden_layer_0/rnn/while/Merge_2_grad/Switch
�
ROptimizer/gradients/hidden_layer_0/rnn/while/Merge_2_grad/tuple/control_dependencyIdentity@Optimizer/gradients/hidden_layer_0/rnn/while/Merge_2_grad/SwitchK^Optimizer/gradients/hidden_layer_0/rnn/while/Merge_2_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Switch_2_grad/b_switch
�
TOptimizer/gradients/hidden_layer_0/rnn/while/Merge_2_grad/tuple/control_dependency_1IdentityBOptimizer/gradients/hidden_layer_0/rnn/while/Merge_2_grad/Switch:1K^Optimizer/gradients/hidden_layer_0/rnn/while/Merge_2_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Switch_2_grad/b_switch
�
@Optimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/SwitchSwitchCOptimizer/gradients/hidden_layer_0/rnn/while/Switch_3_grad/b_switchOptimizer/gradients/b_count_2*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Switch_3_grad/b_switch
�
JOptimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/tuple/group_depsNoOpA^Optimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/Switch
�
ROptimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/tuple/control_dependencyIdentity@Optimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/SwitchK^Optimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Switch_3_grad/b_switch
�
TOptimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/tuple/control_dependency_1IdentityBOptimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/Switch:1K^Optimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Switch_3_grad/b_switch
�
@Optimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/SwitchSwitchCOptimizer/gradients/hidden_layer_0/rnn/while/Switch_4_grad/b_switchOptimizer/gradients/b_count_2*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Switch_4_grad/b_switch
�
JOptimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/tuple/group_depsNoOpA^Optimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/Switch
�
ROptimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/tuple/control_dependencyIdentity@Optimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/SwitchK^Optimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Switch_4_grad/b_switch
�
TOptimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/tuple/control_dependency_1IdentityBOptimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/Switch:1K^Optimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Switch_4_grad/b_switch
�
>Optimizer/gradients/hidden_layer_0/rnn/while/Enter_2_grad/ExitExitROptimizer/gradients/hidden_layer_0/rnn/while/Merge_2_grad/tuple/control_dependency*
T0
�
>Optimizer/gradients/hidden_layer_0/rnn/while/Enter_3_grad/ExitExitROptimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/tuple/control_dependency*
T0
�
>Optimizer/gradients/hidden_layer_0/rnn/while/Enter_4_grad/ExitExitROptimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/tuple/control_dependency*
T0
�
^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_outIdentityiOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPopV2*
T0
�
dOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/ConstConst*7
_class-
+)loc:@hidden_layer_0/rnn/while/dropout/Mul*
dtype0*
valueB :
���������
�
dOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/f_accStackV2dOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/Const*7
_class-
+)loc:@hidden_layer_0/rnn/while/dropout/Mul*
	elem_type0*

stack_name 
�
dOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/EnterEnterdOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
jOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPushV2StackPushV2dOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/Enter<hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3^Optimizer/gradients/Add*
T0*
swap_memory( 
�
iOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPopV2
StackPopV2oOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
oOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPopV2/EnterEnterdOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
eOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/b_syncControlTriggerM^Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPopV2Q^Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/StackPopV2Q^Optimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/StackPopV2s^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2j^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPopV2_^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2a^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1M^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/StackPopV2O^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/StackPopV2c^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2Q^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/StackPopV2U^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2W^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2c^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2e^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1a^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2c^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1U^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2c^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2e^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1Q^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2S^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2c^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2e^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1Q^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2S^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2a^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2c^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1Q^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2
�
cOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrierIdentityTOptimizer/gradients/hidden_layer_0/rnn/while/Merge_2_grad/tuple/control_dependency_1_^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out*
T0
�
wOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3}Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EntercOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrier*7
_class-
+)loc:@hidden_layer_0/rnn/while/dropout/Mul*
sourceOptimizer/gradients
�
}Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterhidden_layer_0/rnn/TensorArray*
T0*7
_class-
+)loc:@hidden_layer_0/rnn/while/dropout/Mul*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
sOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentitycOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrierx^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*7
_class-
+)loc:@hidden_layer_0/rnn/while/dropout/Mul
�
gOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3wOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3rOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2sOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0
�
mOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*6
_class,
*(loc:@hidden_layer_0/rnn/while/Identity_1*
dtype0*
valueB :
���������
�
mOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2mOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*6
_class,
*(loc:@hidden_layer_0/rnn/while/Identity_1*
	elem_type0*

stack_name 
�
mOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEntermOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
sOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2mOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter#hidden_layer_0/rnn/while/Identity_1^Optimizer/gradients/Add*
T0*
swap_memory( 
�
rOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2xOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
xOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEntermOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
fOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOph^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3d^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrier
�
nOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentitygOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3g^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*z
_classp
nlloc:@Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
pOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentitycOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrierg^Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*v
_classl
jhloc:@Optimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrier
�
EOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like	ZerosLikePOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/StackPopV2*
T0
�
KOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/ConstConst*6
_class,
*(loc:@hidden_layer_0/rnn/while/Identity_3*
dtype0*
valueB :
���������
�
KOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/f_accStackV2KOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/Const*6
_class,
*(loc:@hidden_layer_0/rnn/while/Identity_3*
	elem_type0*

stack_name 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
QOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/StackPushV2StackPushV2KOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/Enter#hidden_layer_0/rnn/while/Identity_3^Optimizer/gradients/Add*
T0*
swap_memory( 
�
POptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2VOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
VOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/StackPopV2/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
AOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/SelectSelectLOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPopV2TOptimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/tuple/control_dependency_1EOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like*
T0
�
GOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/ConstConst*8
_class.
,*loc:@hidden_layer_0/rnn/while/GreaterEqual*
dtype0*
valueB :
���������
�
GOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/f_accStackV2GOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/Const*8
_class.
,*loc:@hidden_layer_0/rnn/while/GreaterEqual*
	elem_type0
*

stack_name 
�
GOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/EnterEnterGOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
MOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPushV2StackPushV2GOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/Enter%hidden_layer_0/rnn/while/GreaterEqual^Optimizer/gradients/Add*
T0
*
swap_memory( 
�
LOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPopV2
StackPopV2ROptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0

�
ROptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPopV2/EnterEnterGOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
COptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select_1SelectLOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPopV2EOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_likeTOptimizer/gradients/hidden_layer_0/rnn/while/Merge_3_grad/tuple/control_dependency_1*
T0
�
KOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/tuple/group_depsNoOpB^Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/SelectD^Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select_1
�
SOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/tuple/control_dependencyIdentityAOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/SelectL^Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select
�
UOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/tuple/control_dependency_1IdentityCOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select_1L^Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select_1
�
EOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like	ZerosLikePOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/StackPopV2*
T0
�
KOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/ConstConst*6
_class,
*(loc:@hidden_layer_0/rnn/while/Identity_4*
dtype0*
valueB :
���������
�
KOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/f_accStackV2KOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/Const*6
_class,
*(loc:@hidden_layer_0/rnn/while/Identity_4*
	elem_type0*

stack_name 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
QOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/StackPushV2StackPushV2KOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/Enter#hidden_layer_0/rnn/while/Identity_4^Optimizer/gradients/Add*
T0*
swap_memory( 
�
POptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2VOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
VOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/StackPopV2/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
AOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/SelectSelectLOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPopV2TOptimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/tuple/control_dependency_1EOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like*
T0
�
COptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/Select_1SelectLOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPopV2EOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_likeTOptimizer/gradients/hidden_layer_0/rnn/while/Merge_4_grad/tuple/control_dependency_1*
T0
�
KOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/tuple/group_depsNoOpB^Optimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/SelectD^Optimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/Select_1
�
SOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/tuple/control_dependencyIdentityAOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/SelectL^Optimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/Select
�
UOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/tuple/control_dependency_1IdentityCOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/Select_1L^Optimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/Select_1
�
COptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/zeros_like	ZerosLikeIOptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/zeros_like/Enter^Optimizer/gradients/Sub*
T0
�
IOptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/zeros_like/EnterEnterhidden_layer_0/rnn/zeros*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
?Optimizer/gradients/hidden_layer_0/rnn/while/Select_grad/SelectSelectLOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPopV2nOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyCOptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/zeros_like*
T0
�
AOptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/Select_1SelectLOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select/StackPopV2COptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/zeros_likenOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0
�
IOptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/tuple/group_depsNoOp@^Optimizer/gradients/hidden_layer_0/rnn/while/Select_grad/SelectB^Optimizer/gradients/hidden_layer_0/rnn/while/Select_grad/Select_1
�
QOptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/tuple/control_dependencyIdentity?Optimizer/gradients/hidden_layer_0/rnn/while/Select_grad/SelectJ^Optimizer/gradients/hidden_layer_0/rnn/while/Select_grad/tuple/group_deps*
T0*R
_classH
FDloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Select_grad/Select
�
SOptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/tuple/control_dependency_1IdentityAOptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/Select_1J^Optimizer/gradients/hidden_layer_0/rnn/while/Select_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Select_grad/Select_1
�
COptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/ShapeShape(hidden_layer_0/rnn/while/dropout/RealDiv*
T0*
out_type0
�
EOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Shape_1Shape%hidden_layer_0/rnn/while/dropout/Cast*
T0*
out_type0
�
SOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2`Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
YOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/ConstConst*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Shape*
dtype0*
valueB :
���������
�
YOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_accStackV2YOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/Const*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Shape*
	elem_type0*

stack_name 
�
YOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/EnterEnterYOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
_Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2YOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/EnterCOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Shape^Optimizer/gradients/Add*
T0*
swap_memory( 
�
^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2dOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
dOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnterYOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
[Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/Const_1Const*X
_classN
LJloc:@Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Shape_1*
dtype0*
valueB :
���������
�
[Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_acc_1StackV2[Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/Const_1*X
_classN
LJloc:@Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Shape_1*
	elem_type0*

stack_name 
�
[Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/Enter_1Enter[Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_acc_1*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
aOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2[Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/Enter_1EOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Shape_1^Optimizer/gradients/Add*
T0*
swap_memory( 
�
`Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2fOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer/gradients/Sub*
	elem_type0
�
fOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter[Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_acc_1*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
AOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/MulMulSOptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/tuple/control_dependency_1LOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/StackPopV2*
T0
�
GOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/ConstConst*8
_class.
,*loc:@hidden_layer_0/rnn/while/dropout/Cast*
dtype0*
valueB :
���������
�
GOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/f_accStackV2GOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/Const*8
_class.
,*loc:@hidden_layer_0/rnn/while/dropout/Cast*
	elem_type0*

stack_name 
�
GOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/EnterEnterGOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
MOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/StackPushV2StackPushV2GOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/Enter%hidden_layer_0/rnn/while/dropout/Cast^Optimizer/gradients/Add*
T0*
swap_memory( 
�
LOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/StackPopV2
StackPopV2ROptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
ROptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/StackPopV2/EnterEnterGOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
AOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/SumSumAOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/MulSOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
EOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/ReshapeReshapeAOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Sum^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
COptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1MulNOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/StackPopV2SOptimizer/gradients/hidden_layer_0/rnn/while/Select_grad/tuple/control_dependency_1*
T0
�
IOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/ConstConst*;
_class1
/-loc:@hidden_layer_0/rnn/while/dropout/RealDiv*
dtype0*
valueB :
���������
�
IOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/f_accStackV2IOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/Const*;
_class1
/-loc:@hidden_layer_0/rnn/while/dropout/RealDiv*
	elem_type0*

stack_name 
�
IOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/EnterEnterIOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
OOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/StackPushV2StackPushV2IOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/Enter(hidden_layer_0/rnn/while/dropout/RealDiv^Optimizer/gradients/Add*
T0*
swap_memory( 
�
NOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/StackPopV2
StackPopV2TOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
TOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/StackPopV2/EnterEnterIOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
COptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Sum_1SumCOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Mul_1UOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
GOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Reshape_1ReshapeCOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Sum_1`Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
NOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/tuple/group_depsNoOpF^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/ReshapeH^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Reshape_1
�
VOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/tuple/control_dependencyIdentityEOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/ReshapeO^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Reshape
�
XOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/tuple/control_dependency_1IdentityGOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Reshape_1O^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/Reshape_1
�
JOptimizer/gradients/hidden_layer_0/rnn/while/Switch_2_grad_1/NextIterationNextIterationpOptimizer/gradients/hidden_layer_0/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0
�
GOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/ShapeShape(hidden_layer_0/rnn/while/lstm_cell/mul_2*
T0*
out_type0
�
IOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Shape_1Const^Optimizer/gradients/Sub*
dtype0*
valueB 
�
WOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgsBroadcastGradientArgsbOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2IOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Shape_1*
T0
�
]Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/ConstConst*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Shape*
dtype0*
valueB :
���������
�
]Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/f_accStackV2]Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/Const*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Shape*
	elem_type0*

stack_name 
�
]Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/EnterEnter]Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
cOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPushV2StackPushV2]Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/EnterGOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Shape^Optimizer/gradients/Add*
T0*
swap_memory( 
�
bOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2
StackPopV2hOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
hOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2/EnterEnter]Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
IOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDivRealDivVOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/tuple/control_dependencyTOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2*
T0
�
OOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/ConstConst*7
_class-
+)loc:@hidden_layer_0/rnn/while/dropout/Sub*
dtype0*
valueB :
���������
�
OOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/f_accStackV2OOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/Const*7
_class-
+)loc:@hidden_layer_0/rnn/while/dropout/Sub*
	elem_type0*

stack_name 
�
OOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/EnterEnterOOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
UOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/StackPushV2StackPushV2OOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/Enter$hidden_layer_0/rnn/while/dropout/Sub^Optimizer/gradients/Add*
T0*
swap_memory( 
�
TOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2
StackPopV2ZOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
ZOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2/EnterEnterOOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
EOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/SumSumIOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDivWOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
IOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/ReshapeReshapeEOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/SumbOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
EOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/NegNegPOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/StackPopV2*
T0
�
KOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/ConstConst*;
_class1
/-loc:@hidden_layer_0/rnn/while/lstm_cell/mul_2*
dtype0*
valueB :
���������
�
KOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/f_accStackV2KOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/Const*;
_class1
/-loc:@hidden_layer_0/rnn/while/lstm_cell/mul_2*
	elem_type0*

stack_name 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
QOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/StackPushV2StackPushV2KOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/Enter(hidden_layer_0/rnn/while/lstm_cell/mul_2^Optimizer/gradients/Add*
T0*
swap_memory( 
�
POptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/StackPopV2
StackPopV2VOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
VOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/StackPopV2/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Neg/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv_1RealDivEOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/NegTOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2*
T0
�
KOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv_2RealDivKOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv_1TOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2*
T0
�
EOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/mulMulVOptimizer/gradients/hidden_layer_0/rnn/while/dropout/Mul_grad/tuple/control_dependencyKOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/RealDiv_2*
T0
�
GOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Sum_1SumEOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/mulYOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Reshape_1ReshapeGOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Sum_1IOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Shape_1*
T0*
Tshape0
�
ROptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/tuple/group_depsNoOpJ^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/ReshapeL^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Reshape_1
�
ZOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/tuple/control_dependencyIdentityIOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/ReshapeS^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Reshape
�
\Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/tuple/control_dependency_1IdentityKOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Reshape_1S^Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/tuple/group_deps*
T0*^
_classT
RPloc:@Optimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/Reshape_1
�
Optimizer/gradients/AddN_4AddNUOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/tuple/control_dependency_1ZOptimizer/gradients/hidden_layer_0/rnn/while/dropout/RealDiv_grad/tuple/control_dependency*
N*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/Select_1
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/ShapeShape,hidden_layer_0/rnn/while/lstm_cell/Sigmoid_2*
T0*
out_type0
�
IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Shape_1Shape)hidden_layer_0/rnn/while/lstm_cell/Tanh_1*
T0*
out_type0
�
WOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsbOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2dOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Shape*
dtype0*
valueB :
���������
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Shape*
	elem_type0*

stack_name 
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnter]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
cOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterGOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Shape^Optimizer/gradients/Add*
T0*
swap_memory( 
�
bOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2hOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
hOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnter]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Shape_1*
dtype0*
valueB :
���������
�
_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Shape_1*
	elem_type0*

stack_name 
�
_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1Enter_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
eOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Shape_1^Optimizer/gradients/Add*
T0*
swap_memory( 
�
dOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2jOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer/gradients/Sub*
	elem_type0
�
jOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
EOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/MulMulOptimizer/gradients/AddN_4POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/ConstConst*<
_class2
0.loc:@hidden_layer_0/rnn/while/lstm_cell/Tanh_1*
dtype0*
valueB :
���������
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/f_accStackV2KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/Const*<
_class2
0.loc:@hidden_layer_0/rnn/while/lstm_cell/Tanh_1*
	elem_type0*

stack_name 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/Enter)hidden_layer_0/rnn/while/lstm_cell/Tanh_1^Optimizer/gradients/Add*
T0*
swap_memory( 
�
POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2VOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
VOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
EOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/SumSumEOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/MulWOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/ReshapeReshapeEOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/SumbOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1MulROptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2Optimizer/gradients/AddN_4*
T0
�
MOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*?
_class5
31loc:@hidden_layer_0/rnn/while/lstm_cell/Sigmoid_2*
dtype0*
valueB :
���������
�
MOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2MOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/Const*?
_class5
31loc:@hidden_layer_0/rnn/while/lstm_cell/Sigmoid_2*
	elem_type0*

stack_name 
�
MOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterMOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
SOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2MOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/Enter,hidden_layer_0/rnn/while/lstm_cell/Sigmoid_2^Optimizer/gradients/Add*
T0*
swap_memory( 
�
ROptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2XOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
XOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterMOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Sum_1SumGOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1YOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Reshape_1ReshapeGOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Sum_1dOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
ROptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/tuple/group_depsNoOpJ^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/ReshapeL^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Reshape_1
�
ZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityIOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/ReshapeS^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Reshape
�
\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityKOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Reshape_1S^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*^
_classT
RPloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Reshape_1
�
QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradROptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2ZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency*
T0
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradPOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0
�
Optimizer/gradients/AddN_5AddNUOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/tuple/control_dependency_1KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/Tanh_1_grad/TanhGrad*
N*
T0*V
_classL
JHloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select_1
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/ShapeShape&hidden_layer_0/rnn/while/lstm_cell/mul*
T0*
out_type0
�
IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Shape_1Shape(hidden_layer_0/rnn/while/lstm_cell/mul_1*
T0*
out_type0
�
WOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsbOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2dOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Shape*
dtype0*
valueB :
���������
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Shape*
	elem_type0*

stack_name 
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnter]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
cOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterGOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Shape^Optimizer/gradients/Add*
T0*
swap_memory( 
�
bOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2hOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
hOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Shape_1*
dtype0*
valueB :
���������
�
_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Shape_1*
	elem_type0*

stack_name 
�
_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1Enter_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
eOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Shape_1^Optimizer/gradients/Add*
T0*
swap_memory( 
�
dOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2jOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer/gradients/Sub*
	elem_type0
�
jOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
EOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/SumSumOptimizer/gradients/AddN_5WOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/ReshapeReshapeEOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/SumbOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Sum_1SumOptimizer/gradients/AddN_5YOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Reshape_1ReshapeGOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Sum_1dOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
ROptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/tuple/group_depsNoOpJ^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/ReshapeL^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Reshape_1
�
ZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentityIOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/ReshapeS^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Reshape
�
\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityKOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Reshape_1S^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/Reshape_1
�
EOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/ShapeShape*hidden_layer_0/rnn/while/lstm_cell/Sigmoid*
T0*
out_type0
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Shape_1Shape#hidden_layer_0/rnn/while/Identity_3*
T0*
out_type0
�
UOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgs`Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2bOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*X
_classN
LJloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Shape*
dtype0*
valueB :
���������
�
[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*X
_classN
LJloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Shape*
	elem_type0*

stack_name 
�
[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnter[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
aOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Shape^Optimizer/gradients/Add*
T0*
swap_memory( 
�
`Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2fOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
fOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Shape_1*
dtype0*
valueB :
���������
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Shape_1*
	elem_type0*

stack_name 
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1Enter]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
cOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Shape_1^Optimizer/gradients/Add*
T0*
swap_memory( 
�
bOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2hOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer/gradients/Sub*
	elem_type0
�
hOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
COptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/MulMulZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyPOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/zeros_like/StackPopV2*
T0
�
COptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/SumSumCOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/MulUOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/ReshapeReshapeCOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Sum`Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
EOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1MulPOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2ZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency*
T0
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/ConstConst*=
_class3
1/loc:@hidden_layer_0/rnn/while/lstm_cell/Sigmoid*
dtype0*
valueB :
���������
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/f_accStackV2KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/Const*=
_class3
1/loc:@hidden_layer_0/rnn/while/lstm_cell/Sigmoid*
	elem_type0*

stack_name 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/Enter*hidden_layer_0/rnn/while/lstm_cell/Sigmoid^Optimizer/gradients/Add*
T0*
swap_memory( 
�
POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2VOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
VOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
EOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Sum_1SumEOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1WOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Reshape_1ReshapeEOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Sum_1bOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/tuple/group_depsNoOpH^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/ReshapeJ^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Reshape_1
�
XOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/tuple/control_dependencyIdentityGOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/ReshapeQ^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Reshape
�
ZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1IdentityIOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Reshape_1Q^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Reshape_1
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/ShapeShape,hidden_layer_0/rnn/while/lstm_cell/Sigmoid_1*
T0*
out_type0
�
IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Shape_1Shape'hidden_layer_0/rnn/while/lstm_cell/Tanh*
T0*
out_type0
�
WOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsbOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2dOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Shape*
dtype0*
valueB :
���������
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Shape*
	elem_type0*

stack_name 
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnter]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
cOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterGOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Shape^Optimizer/gradients/Add*
T0*
swap_memory( 
�
bOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2hOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
hOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnter]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Shape_1*
dtype0*
valueB :
���������
�
_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Shape_1*
	elem_type0*

stack_name 
�
_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1Enter_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
eOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Shape_1^Optimizer/gradients/Add*
T0*
swap_memory( 
�
dOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2jOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer/gradients/Sub*
	elem_type0
�
jOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter_Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
EOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/MulMul\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/ConstConst*:
_class0
.,loc:@hidden_layer_0/rnn/while/lstm_cell/Tanh*
dtype0*
valueB :
���������
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/f_accStackV2KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/Const*:
_class0
.,loc:@hidden_layer_0/rnn/while/lstm_cell/Tanh*
	elem_type0*

stack_name 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/Enter'hidden_layer_0/rnn/while/lstm_cell/Tanh^Optimizer/gradients/Add*
T0*
swap_memory( 
�
POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2VOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
VOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterKOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
EOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/SumSumEOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/MulWOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/ReshapeReshapeEOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/SumbOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1MulROptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0
�
MOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*?
_class5
31loc:@hidden_layer_0/rnn/while/lstm_cell/Sigmoid_1*
dtype0*
valueB :
���������
�
MOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2MOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/Const*?
_class5
31loc:@hidden_layer_0/rnn/while/lstm_cell/Sigmoid_1*
	elem_type0*

stack_name 
�
MOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterMOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
SOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2MOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/Enter,hidden_layer_0/rnn/while/lstm_cell/Sigmoid_1^Optimizer/gradients/Add*
T0*
swap_memory( 
�
ROptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2XOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
XOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterMOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Sum_1SumGOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1YOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Reshape_1ReshapeGOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Sum_1dOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
ROptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/tuple/group_depsNoOpJ^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/ReshapeL^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Reshape_1
�
ZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityIOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/ReshapeS^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Reshape
�
\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityKOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Reshape_1S^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Reshape_1
�
OOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradPOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2XOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/tuple/control_dependency*
T0
�
Optimizer/gradients/AddN_6AddNSOptimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/tuple/control_dependencyZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1*
N*
T0*T
_classJ
HFloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Select_1_grad/Select
�
QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradROptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2ZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency*
T0
�
IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/Tanh_grad/TanhGradTanhGradPOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0
�
EOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/ShapeShape*hidden_layer_0/rnn/while/lstm_cell/split:2*
T0*
out_type0
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Shape_1Shape(hidden_layer_0/rnn/while/lstm_cell/add/y*
T0*
out_type0
�
UOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgs`Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2bOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*X
_classN
LJloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Shape*
dtype0*
valueB :
���������
�
[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*X
_classN
LJloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Shape*
	elem_type0*

stack_name 
�
[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnter[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
aOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Shape^Optimizer/gradients/Add*
T0*
swap_memory( 
�
`Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2fOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
fOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnter[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Const_1Const*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Shape_1*
dtype0*
valueB :
���������
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc_1StackV2]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Const_1*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Shape_1*
	elem_type0*

stack_name 
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter_1Enter]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc_1*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
cOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter_1GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Shape_1^Optimizer/gradients/Add*
T0*
swap_memory( 
�
bOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2hOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer/gradients/Sub*
	elem_type0
�
hOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc_1*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
COptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/SumSumOOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradUOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/ReshapeReshapeCOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Sum`Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
EOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Sum_1SumOOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradWOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Reshape_1ReshapeEOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Sum_1bOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/tuple/group_depsNoOpH^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/ReshapeJ^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Reshape_1
�
XOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/tuple/control_dependencyIdentityGOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/ReshapeQ^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Reshape
�
ZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/tuple/control_dependency_1IdentityIOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Reshape_1Q^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/Reshape_1
�
JOptimizer/gradients/hidden_layer_0/rnn/while/Switch_3_grad_1/NextIterationNextIterationOptimizer/gradients/AddN_6*
T0
�
HOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/split_grad/concatConcatV2QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGradIOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/Tanh_grad/TanhGradXOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/add_grad/tuple/control_dependencyQOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradNOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/split_grad/concat/Const*
N*
T0*

Tidx0
�
NOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/split_grad/concat/ConstConst^Optimizer/gradients/Sub*
dtype0*
value	B :
�
OOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradHOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/split_grad/concat*
T0*
data_formatNHWC
�
TOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpP^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradI^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/split_grad/concat
�
\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityHOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/split_grad/concatU^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/split_grad/concat
�
^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentityOOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradU^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*b
_classX
VTloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGrad
�
IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMulMatMul\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyOOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul/Enter*
T0*
transpose_a( *
transpose_b(
�
OOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul/EnterEnter(hidden_layer_0/rnn/lstm_cell/kernel/read*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1MatMulVOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*<
_class2
0.loc:@hidden_layer_0/rnn/while/lstm_cell/concat*
dtype0*
valueB :
���������
�
QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Const*<
_class2
0.loc:@hidden_layer_0/rnn/while/lstm_cell/concat*
	elem_type0*

stack_name 
�
QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterQOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
WOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Enter)hidden_layer_0/rnn/while/lstm_cell/concat^Optimizer/gradients/Add*
T0*
swap_memory( 
�
VOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
\Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterQOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
SOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/tuple/group_depsNoOpJ^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMulL^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1
�
[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityIOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMulT^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityKOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1T^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/MatMul_1
�
OOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
valueB�*    
�
QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterOOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeQOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1WOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
T0
�
POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchQOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2Optimizer/gradients/b_count_2*
T0
�
MOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/AddAddV2ROptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch:1^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0
�
WOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationMOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/Add*
T0
�
QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitPOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch*
T0
�
HOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ConstConst^Optimizer/gradients/Sub*
dtype0*
value	B :
�
GOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/RankConst^Optimizer/gradients/Sub*
dtype0*
value	B :
�
FOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/modFloorModHOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ConstGOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/Rank*
T0
�
HOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeShape*hidden_layer_0/rnn/while/TensorArrayReadV3*
T0*
out_type0
�
IOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeNShapeNTOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2POptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/zeros_like/StackPopV2*
N*
T0*
out_type0
�
OOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/ConstConst*=
_class3
1/loc:@hidden_layer_0/rnn/while/TensorArrayReadV3*
dtype0*
valueB :
���������
�
OOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/f_accStackV2OOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/Const*=
_class3
1/loc:@hidden_layer_0/rnn/while/TensorArrayReadV3*
	elem_type0*

stack_name 
�
OOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/EnterEnterOOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*6

frame_name(&hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
UOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2OOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/Enter*hidden_layer_0/rnn/while/TensorArrayReadV3^Optimizer/gradients/Add*
T0*
swap_memory( 
�
TOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2ZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^Optimizer/gradients/Sub*
	elem_type0
�
ZOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterOOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
OOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ConcatOffsetConcatOffsetFOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/modIOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeNKOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN:1*
N
�
HOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/SliceSlice[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyOOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ConcatOffsetIOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN*
Index0*
T0
�
JOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/Slice_1Slice[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyQOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ConcatOffset:1KOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0
�
SOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/tuple/group_depsNoOpI^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/SliceK^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/Slice_1
�
[Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/tuple/control_dependencyIdentityHOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/SliceT^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/tuple/group_deps*
T0*[
_classQ
OMloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/Slice
�
]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1IdentityJOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/Slice_1T^Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/tuple/group_deps*
T0*]
_classS
QOloc:@Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/Slice_1
�
NOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/b_accConst*
dtype0*
valueB
��*    
�
POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterNOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc*
T0*J

frame_name<:Optimizer/gradients/hidden_layer_0/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergePOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1VOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/NextIteration*
N*
T0
�
OOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchPOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2Optimizer/gradients/b_count_2*
T0
�
LOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/AddAddV2QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/Switch:1]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0
�
VOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationLOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/Add*
T0
�
POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitOOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/Switch*
T0
�
Optimizer/gradients/AddN_7AddNSOptimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/tuple/control_dependency]Optimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1*
N*
T0*T
_classJ
HFloc:@Optimizer/gradients/hidden_layer_0/rnn/while/Select_2_grad/Select
�
JOptimizer/gradients/hidden_layer_0/rnn/while/Switch_4_grad_1/NextIterationNextIterationOptimizer/gradients/AddN_7*
T0
�
Optimizer/clip_by_norm/mulMulPOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3POptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
Q
Optimizer/clip_by_norm/ConstConst*
dtype0*
valueB"       
�
Optimizer/clip_by_norm/SumSumOptimizer/clip_by_norm/mulOptimizer/clip_by_norm/Const*
T0*

Tidx0*
	keep_dims(
M
 Optimizer/clip_by_norm/Greater/yConst*
dtype0*
valueB
 *    
p
Optimizer/clip_by_norm/GreaterGreaterOptimizer/clip_by_norm/Sum Optimizer/clip_by_norm/Greater/y*
T0
[
&Optimizer/clip_by_norm/ones_like/ShapeConst*
dtype0*
valueB"      
S
&Optimizer/clip_by_norm/ones_like/ConstConst*
dtype0*
valueB
 *  �?
�
 Optimizer/clip_by_norm/ones_likeFill&Optimizer/clip_by_norm/ones_like/Shape&Optimizer/clip_by_norm/ones_like/Const*
T0*

index_type0
�
Optimizer/clip_by_norm/SelectSelectOptimizer/clip_by_norm/GreaterOptimizer/clip_by_norm/Sum Optimizer/clip_by_norm/ones_like*
T0
K
Optimizer/clip_by_norm/SqrtSqrtOptimizer/clip_by_norm/Select*
T0
�
Optimizer/clip_by_norm/Select_1SelectOptimizer/clip_by_norm/GreaterOptimizer/clip_by_norm/SqrtOptimizer/clip_by_norm/Sum*
T0
K
Optimizer/clip_by_norm/mul_1/yConst*
dtype0*
valueB
 *  �@
�
Optimizer/clip_by_norm/mul_1MulPOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3Optimizer/clip_by_norm/mul_1/y*
T0
M
 Optimizer/clip_by_norm/Maximum/yConst*
dtype0*
valueB
 *  �@
u
Optimizer/clip_by_norm/MaximumMaximumOptimizer/clip_by_norm/Select_1 Optimizer/clip_by_norm/Maximum/y*
T0
p
Optimizer/clip_by_norm/truedivRealDivOptimizer/clip_by_norm/mul_1Optimizer/clip_by_norm/Maximum*
T0
K
Optimizer/clip_by_normIdentityOptimizer/clip_by_norm/truediv*
T0
�
Optimizer/clip_by_norm_1/mulMulQOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3QOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
L
Optimizer/clip_by_norm_1/ConstConst*
dtype0*
valueB: 
�
Optimizer/clip_by_norm_1/SumSumOptimizer/clip_by_norm_1/mulOptimizer/clip_by_norm_1/Const*
T0*

Tidx0*
	keep_dims(
O
"Optimizer/clip_by_norm_1/Greater/yConst*
dtype0*
valueB
 *    
v
 Optimizer/clip_by_norm_1/GreaterGreaterOptimizer/clip_by_norm_1/Sum"Optimizer/clip_by_norm_1/Greater/y*
T0
V
(Optimizer/clip_by_norm_1/ones_like/ShapeConst*
dtype0*
valueB:
U
(Optimizer/clip_by_norm_1/ones_like/ConstConst*
dtype0*
valueB
 *  �?
�
"Optimizer/clip_by_norm_1/ones_likeFill(Optimizer/clip_by_norm_1/ones_like/Shape(Optimizer/clip_by_norm_1/ones_like/Const*
T0*

index_type0
�
Optimizer/clip_by_norm_1/SelectSelect Optimizer/clip_by_norm_1/GreaterOptimizer/clip_by_norm_1/Sum"Optimizer/clip_by_norm_1/ones_like*
T0
O
Optimizer/clip_by_norm_1/SqrtSqrtOptimizer/clip_by_norm_1/Select*
T0
�
!Optimizer/clip_by_norm_1/Select_1Select Optimizer/clip_by_norm_1/GreaterOptimizer/clip_by_norm_1/SqrtOptimizer/clip_by_norm_1/Sum*
T0
M
 Optimizer/clip_by_norm_1/mul_1/yConst*
dtype0*
valueB
 *  �@
�
Optimizer/clip_by_norm_1/mul_1MulQOptimizer/gradients/hidden_layer_0/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3 Optimizer/clip_by_norm_1/mul_1/y*
T0
O
"Optimizer/clip_by_norm_1/Maximum/yConst*
dtype0*
valueB
 *  �@
{
 Optimizer/clip_by_norm_1/MaximumMaximum!Optimizer/clip_by_norm_1/Select_1"Optimizer/clip_by_norm_1/Maximum/y*
T0
v
 Optimizer/clip_by_norm_1/truedivRealDivOptimizer/clip_by_norm_1/mul_1 Optimizer/clip_by_norm_1/Maximum*
T0
O
Optimizer/clip_by_norm_1Identity Optimizer/clip_by_norm_1/truediv*
T0
�
Optimizer/clip_by_norm_2/mulMulGOptimizer/gradients/output_layer/MatMul_grad/tuple/control_dependency_1GOptimizer/gradients/output_layer/MatMul_grad/tuple/control_dependency_1*
T0
S
Optimizer/clip_by_norm_2/ConstConst*
dtype0*
valueB"       
�
Optimizer/clip_by_norm_2/SumSumOptimizer/clip_by_norm_2/mulOptimizer/clip_by_norm_2/Const*
T0*

Tidx0*
	keep_dims(
O
"Optimizer/clip_by_norm_2/Greater/yConst*
dtype0*
valueB
 *    
v
 Optimizer/clip_by_norm_2/GreaterGreaterOptimizer/clip_by_norm_2/Sum"Optimizer/clip_by_norm_2/Greater/y*
T0
]
(Optimizer/clip_by_norm_2/ones_like/ShapeConst*
dtype0*
valueB"      
U
(Optimizer/clip_by_norm_2/ones_like/ConstConst*
dtype0*
valueB
 *  �?
�
"Optimizer/clip_by_norm_2/ones_likeFill(Optimizer/clip_by_norm_2/ones_like/Shape(Optimizer/clip_by_norm_2/ones_like/Const*
T0*

index_type0
�
Optimizer/clip_by_norm_2/SelectSelect Optimizer/clip_by_norm_2/GreaterOptimizer/clip_by_norm_2/Sum"Optimizer/clip_by_norm_2/ones_like*
T0
O
Optimizer/clip_by_norm_2/SqrtSqrtOptimizer/clip_by_norm_2/Select*
T0
�
!Optimizer/clip_by_norm_2/Select_1Select Optimizer/clip_by_norm_2/GreaterOptimizer/clip_by_norm_2/SqrtOptimizer/clip_by_norm_2/Sum*
T0
M
 Optimizer/clip_by_norm_2/mul_1/yConst*
dtype0*
valueB
 *  �@
�
Optimizer/clip_by_norm_2/mul_1MulGOptimizer/gradients/output_layer/MatMul_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_2/mul_1/y*
T0
O
"Optimizer/clip_by_norm_2/Maximum/yConst*
dtype0*
valueB
 *  �@
{
 Optimizer/clip_by_norm_2/MaximumMaximum!Optimizer/clip_by_norm_2/Select_1"Optimizer/clip_by_norm_2/Maximum/y*
T0
v
 Optimizer/clip_by_norm_2/truedivRealDivOptimizer/clip_by_norm_2/mul_1 Optimizer/clip_by_norm_2/Maximum*
T0
O
Optimizer/clip_by_norm_2Identity Optimizer/clip_by_norm_2/truediv*
T0
�
Optimizer/clip_by_norm_3/mulMulDOptimizer/gradients/output_layer/add_grad/tuple/control_dependency_1DOptimizer/gradients/output_layer/add_grad/tuple/control_dependency_1*
T0
L
Optimizer/clip_by_norm_3/ConstConst*
dtype0*
valueB: 
�
Optimizer/clip_by_norm_3/SumSumOptimizer/clip_by_norm_3/mulOptimizer/clip_by_norm_3/Const*
T0*

Tidx0*
	keep_dims(
O
"Optimizer/clip_by_norm_3/Greater/yConst*
dtype0*
valueB
 *    
v
 Optimizer/clip_by_norm_3/GreaterGreaterOptimizer/clip_by_norm_3/Sum"Optimizer/clip_by_norm_3/Greater/y*
T0
V
(Optimizer/clip_by_norm_3/ones_like/ShapeConst*
dtype0*
valueB:
U
(Optimizer/clip_by_norm_3/ones_like/ConstConst*
dtype0*
valueB
 *  �?
�
"Optimizer/clip_by_norm_3/ones_likeFill(Optimizer/clip_by_norm_3/ones_like/Shape(Optimizer/clip_by_norm_3/ones_like/Const*
T0*

index_type0
�
Optimizer/clip_by_norm_3/SelectSelect Optimizer/clip_by_norm_3/GreaterOptimizer/clip_by_norm_3/Sum"Optimizer/clip_by_norm_3/ones_like*
T0
O
Optimizer/clip_by_norm_3/SqrtSqrtOptimizer/clip_by_norm_3/Select*
T0
�
!Optimizer/clip_by_norm_3/Select_1Select Optimizer/clip_by_norm_3/GreaterOptimizer/clip_by_norm_3/SqrtOptimizer/clip_by_norm_3/Sum*
T0
M
 Optimizer/clip_by_norm_3/mul_1/yConst*
dtype0*
valueB
 *  �@
�
Optimizer/clip_by_norm_3/mul_1MulDOptimizer/gradients/output_layer/add_grad/tuple/control_dependency_1 Optimizer/clip_by_norm_3/mul_1/y*
T0
O
"Optimizer/clip_by_norm_3/Maximum/yConst*
dtype0*
valueB
 *  �@
{
 Optimizer/clip_by_norm_3/MaximumMaximum!Optimizer/clip_by_norm_3/Select_1"Optimizer/clip_by_norm_3/Maximum/y*
T0
v
 Optimizer/clip_by_norm_3/truedivRealDivOptimizer/clip_by_norm_3/mul_1 Optimizer/clip_by_norm_3/Maximum*
T0
O
Optimizer/clip_by_norm_3Identity Optimizer/clip_by_norm_3/truediv*
T0
�
#Optimizer/beta1_power/initial_valueConst*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
dtype0*
valueB
 *fff?
�
Optimizer/beta1_power
VariableV2*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
	container *
dtype0*
shape: *
shared_name 
�
Optimizer/beta1_power/AssignAssignOptimizer/beta1_power#Optimizer/beta1_power/initial_value*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
|
Optimizer/beta1_power/readIdentityOptimizer/beta1_power*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias
�
#Optimizer/beta2_power/initial_valueConst*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
dtype0*
valueB
 *w�?
�
Optimizer/beta2_power
VariableV2*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
	container *
dtype0*
shape: *
shared_name 
�
Optimizer/beta2_power/AssignAssignOptimizer/beta2_power#Optimizer/beta2_power/initial_value*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
|
Optimizer/beta2_power/readIdentityOptimizer/beta2_power*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias
�
TOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorConst*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
dtype0*
valueB"�  �  
�
JOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam/Initializer/zeros/ConstConst*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
dtype0*
valueB
 *    
�
DOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam/Initializer/zerosFillTOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam/Initializer/zeros/shape_as_tensorJOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam/Initializer/zeros/Const*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*

index_type0
�
2Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam
VariableV2*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
	container *
dtype0*
shape:
��*
shared_name 
�
9Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam/AssignAssign2Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/AdamDOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam/Initializer/zeros*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
use_locking(*
validate_shape(
�
7Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam/readIdentity2Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel
�
VOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
dtype0*
valueB"�  �  
�
LOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/ConstConst*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
dtype0*
valueB
 *    
�
FOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1/Initializer/zerosFillVOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/shape_as_tensorLOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros/Const*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*

index_type0
�
4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1
VariableV2*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
	container *
dtype0*
shape:
��*
shared_name 
�
;Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1/AssignAssign4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1FOptimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1/Initializer/zeros*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
use_locking(*
validate_shape(
�
9Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1/readIdentity4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel
�
BOptimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam/Initializer/zerosConst*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
dtype0*
valueB�*    
�
0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam
VariableV2*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
	container *
dtype0*
shape:�*
shared_name 
�
7Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam/AssignAssign0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/AdamBOptimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam/Initializer/zeros*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
5Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam/readIdentity0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias
�
DOptimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1/Initializer/zerosConst*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
dtype0*
valueB�*    
�
2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1
VariableV2*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
	container *
dtype0*
shape:�*
shared_name 
�
9Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1/AssignAssign2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1DOptimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1/Initializer/zeros*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
7Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1/readIdentity2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias
�
EOptimizer/output_layer/weights/Adam/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@output_layer/weights*
dtype0*
valueB"f   I  
�
;Optimizer/output_layer/weights/Adam/Initializer/zeros/ConstConst*'
_class
loc:@output_layer/weights*
dtype0*
valueB
 *    
�
5Optimizer/output_layer/weights/Adam/Initializer/zerosFillEOptimizer/output_layer/weights/Adam/Initializer/zeros/shape_as_tensor;Optimizer/output_layer/weights/Adam/Initializer/zeros/Const*
T0*'
_class
loc:@output_layer/weights*

index_type0
�
#Optimizer/output_layer/weights/Adam
VariableV2*'
_class
loc:@output_layer/weights*
	container *
dtype0*
shape:	f�*
shared_name 
�
*Optimizer/output_layer/weights/Adam/AssignAssign#Optimizer/output_layer/weights/Adam5Optimizer/output_layer/weights/Adam/Initializer/zeros*
T0*'
_class
loc:@output_layer/weights*
use_locking(*
validate_shape(
�
(Optimizer/output_layer/weights/Adam/readIdentity#Optimizer/output_layer/weights/Adam*
T0*'
_class
loc:@output_layer/weights
�
GOptimizer/output_layer/weights/Adam_1/Initializer/zeros/shape_as_tensorConst*'
_class
loc:@output_layer/weights*
dtype0*
valueB"f   I  
�
=Optimizer/output_layer/weights/Adam_1/Initializer/zeros/ConstConst*'
_class
loc:@output_layer/weights*
dtype0*
valueB
 *    
�
7Optimizer/output_layer/weights/Adam_1/Initializer/zerosFillGOptimizer/output_layer/weights/Adam_1/Initializer/zeros/shape_as_tensor=Optimizer/output_layer/weights/Adam_1/Initializer/zeros/Const*
T0*'
_class
loc:@output_layer/weights*

index_type0
�
%Optimizer/output_layer/weights/Adam_1
VariableV2*'
_class
loc:@output_layer/weights*
	container *
dtype0*
shape:	f�*
shared_name 
�
,Optimizer/output_layer/weights/Adam_1/AssignAssign%Optimizer/output_layer/weights/Adam_17Optimizer/output_layer/weights/Adam_1/Initializer/zeros*
T0*'
_class
loc:@output_layer/weights*
use_locking(*
validate_shape(
�
*Optimizer/output_layer/weights/Adam_1/readIdentity%Optimizer/output_layer/weights/Adam_1*
T0*'
_class
loc:@output_layer/weights
�
DOptimizer/output_layer/biases/Adam/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@output_layer/biases*
dtype0*
valueB:�
�
:Optimizer/output_layer/biases/Adam/Initializer/zeros/ConstConst*&
_class
loc:@output_layer/biases*
dtype0*
valueB
 *    
�
4Optimizer/output_layer/biases/Adam/Initializer/zerosFillDOptimizer/output_layer/biases/Adam/Initializer/zeros/shape_as_tensor:Optimizer/output_layer/biases/Adam/Initializer/zeros/Const*
T0*&
_class
loc:@output_layer/biases*

index_type0
�
"Optimizer/output_layer/biases/Adam
VariableV2*&
_class
loc:@output_layer/biases*
	container *
dtype0*
shape:�*
shared_name 
�
)Optimizer/output_layer/biases/Adam/AssignAssign"Optimizer/output_layer/biases/Adam4Optimizer/output_layer/biases/Adam/Initializer/zeros*
T0*&
_class
loc:@output_layer/biases*
use_locking(*
validate_shape(
�
'Optimizer/output_layer/biases/Adam/readIdentity"Optimizer/output_layer/biases/Adam*
T0*&
_class
loc:@output_layer/biases
�
FOptimizer/output_layer/biases/Adam_1/Initializer/zeros/shape_as_tensorConst*&
_class
loc:@output_layer/biases*
dtype0*
valueB:�
�
<Optimizer/output_layer/biases/Adam_1/Initializer/zeros/ConstConst*&
_class
loc:@output_layer/biases*
dtype0*
valueB
 *    
�
6Optimizer/output_layer/biases/Adam_1/Initializer/zerosFillFOptimizer/output_layer/biases/Adam_1/Initializer/zeros/shape_as_tensor<Optimizer/output_layer/biases/Adam_1/Initializer/zeros/Const*
T0*&
_class
loc:@output_layer/biases*

index_type0
�
$Optimizer/output_layer/biases/Adam_1
VariableV2*&
_class
loc:@output_layer/biases*
	container *
dtype0*
shape:�*
shared_name 
�
+Optimizer/output_layer/biases/Adam_1/AssignAssign$Optimizer/output_layer/biases/Adam_16Optimizer/output_layer/biases/Adam_1/Initializer/zeros*
T0*&
_class
loc:@output_layer/biases*
use_locking(*
validate_shape(
�
)Optimizer/output_layer/biases/Adam_1/readIdentity$Optimizer/output_layer/biases/Adam_1*
T0*&
_class
loc:@output_layer/biases
I
Optimizer/Adam/learning_rateConst*
dtype0*
valueB
 *j.�;
A
Optimizer/Adam/beta1Const*
dtype0*
valueB
 *fff?
A
Optimizer/Adam/beta2Const*
dtype0*
valueB
 *w�?
C
Optimizer/Adam/epsilonConst*
dtype0*
valueB
 *w�+2
�
COptimizer/Adam/update_hidden_layer_0/rnn/lstm_cell/kernel/ApplyAdam	ApplyAdam#hidden_layer_0/rnn/lstm_cell/kernel2Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readOptimizer/Adam/learning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
use_locking( *
use_nesterov( 
�
AOptimizer/Adam/update_hidden_layer_0/rnn/lstm_cell/bias/ApplyAdam	ApplyAdam!hidden_layer_0/rnn/lstm_cell/bias0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readOptimizer/Adam/learning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_1*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking( *
use_nesterov( 
�
4Optimizer/Adam/update_output_layer/weights/ApplyAdam	ApplyAdamoutput_layer/weights#Optimizer/output_layer/weights/Adam%Optimizer/output_layer/weights/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readOptimizer/Adam/learning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_2*
T0*'
_class
loc:@output_layer/weights*
use_locking( *
use_nesterov( 
�
3Optimizer/Adam/update_output_layer/biases/ApplyAdam	ApplyAdamoutput_layer/biases"Optimizer/output_layer/biases/Adam$Optimizer/output_layer/biases/Adam_1Optimizer/beta1_power/readOptimizer/beta2_power/readOptimizer/Adam/learning_rateOptimizer/Adam/beta1Optimizer/Adam/beta2Optimizer/Adam/epsilonOptimizer/clip_by_norm_3*
T0*&
_class
loc:@output_layer/biases*
use_locking( *
use_nesterov( 
�
Optimizer/Adam/mulMulOptimizer/beta1_power/readOptimizer/Adam/beta1B^Optimizer/Adam/update_hidden_layer_0/rnn/lstm_cell/bias/ApplyAdamD^Optimizer/Adam/update_hidden_layer_0/rnn/lstm_cell/kernel/ApplyAdam4^Optimizer/Adam/update_output_layer/biases/ApplyAdam5^Optimizer/Adam/update_output_layer/weights/ApplyAdam*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias
�
Optimizer/Adam/AssignAssignOptimizer/beta1_powerOptimizer/Adam/mul*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking( *
validate_shape(
�
Optimizer/Adam/mul_1MulOptimizer/beta2_power/readOptimizer/Adam/beta2B^Optimizer/Adam/update_hidden_layer_0/rnn/lstm_cell/bias/ApplyAdamD^Optimizer/Adam/update_hidden_layer_0/rnn/lstm_cell/kernel/ApplyAdam4^Optimizer/Adam/update_output_layer/biases/ApplyAdam5^Optimizer/Adam/update_output_layer/weights/ApplyAdam*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias
�
Optimizer/Adam/Assign_1AssignOptimizer/beta2_powerOptimizer/Adam/mul_1*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking( *
validate_shape(
�
Optimizer/AdamNoOp^Optimizer/Adam/Assign^Optimizer/Adam/Assign_1B^Optimizer/Adam/update_hidden_layer_0/rnn/lstm_cell/bias/ApplyAdamD^Optimizer/Adam/update_hidden_layer_0/rnn/lstm_cell/kernel/ApplyAdam4^Optimizer/Adam/update_output_layer/biases/ApplyAdam5^Optimizer/Adam/update_output_layer/weights/ApplyAdam
�
initNoOp^Optimizer/beta1_power/Assign^Optimizer/beta2_power/Assign8^Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam/Assign:^Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1/Assign:^Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam/Assign<^Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1/Assign*^Optimizer/output_layer/biases/Adam/Assign,^Optimizer/output_layer/biases/Adam_1/Assign+^Optimizer/output_layer/weights/Adam/Assign-^Optimizer/output_layer/weights/Adam_1/Assign)^hidden_layer_0/rnn/lstm_cell/bias/Assign+^hidden_layer_0/rnn/lstm_cell/kernel/Assign^output_layer/biases/Assign^output_layer/weights/Assign
K
X_1Placeholder*
dtype0**
shape!:�������������������
O
y_seq_1Placeholder*
dtype0**
shape!:�������������������
P
y_corr_1Placeholder*
dtype0**
shape!:�������������������
>
keep_prob_1/inputConst*
dtype0*
valueB
 *  �?
R
keep_prob_1PlaceholderWithDefaultkeep_prob_1/input*
dtype0*
shape: 

Abs_1AbsX_1*
T0
A
Max_1/reduction_indicesConst*
dtype0*
value	B :
R
Max_1MaxAbs_1Max_1/reduction_indices*
T0*

Tidx0*
	keep_dims( 

Sign_1SignMax_1*
T0
A
Sum_1/reduction_indicesConst*
dtype0*
value	B :
S
Sum_1SumSign_1Sum_1/reduction_indices*
T0*

Tidx0*
	keep_dims( 
=
Cast_1CastSum_1*

DstT0*

SrcT0*
Truncate( 
V
)hidden_layer_0_1/DropoutWrapperInit/ConstConst*
dtype0*
valueB
 *  �?
X
+hidden_layer_0_1/DropoutWrapperInit/Const_1Const*
dtype0*
valueB
 *  �?
C
hidden_layer_0_1/rnn/RankConst*
dtype0*
value	B :
J
 hidden_layer_0_1/rnn/range/startConst*
dtype0*
value	B :
J
 hidden_layer_0_1/rnn/range/deltaConst*
dtype0*
value	B :
�
hidden_layer_0_1/rnn/rangeRange hidden_layer_0_1/rnn/range/starthidden_layer_0_1/rnn/Rank hidden_layer_0_1/rnn/range/delta*

Tidx0
Y
$hidden_layer_0_1/rnn/concat/values_0Const*
dtype0*
valueB"       
J
 hidden_layer_0_1/rnn/concat/axisConst*
dtype0*
value	B : 
�
hidden_layer_0_1/rnn/concatConcatV2$hidden_layer_0_1/rnn/concat/values_0hidden_layer_0_1/rnn/range hidden_layer_0_1/rnn/concat/axis*
N*
T0*

Tidx0
c
hidden_layer_0_1/rnn/transpose	TransposeX_1hidden_layer_0_1/rnn/concat*
T0*
Tperm0
A
$hidden_layer_0_1/rnn/sequence_lengthIdentityCast_1*
T0
\
hidden_layer_0_1/rnn/ShapeShapehidden_layer_0_1/rnn/transpose*
T0*
out_type0
V
(hidden_layer_0_1/rnn/strided_slice/stackConst*
dtype0*
valueB:
X
*hidden_layer_0_1/rnn/strided_slice/stack_1Const*
dtype0*
valueB:
X
*hidden_layer_0_1/rnn/strided_slice/stack_2Const*
dtype0*
valueB:
�
"hidden_layer_0_1/rnn/strided_sliceStridedSlicehidden_layer_0_1/rnn/Shape(hidden_layer_0_1/rnn/strided_slice/stack*hidden_layer_0_1/rnn/strided_slice/stack_1*hidden_layer_0_1/rnn/strided_slice/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
w
Mhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dimConst*
dtype0*
value	B : 
�
Ihidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims
ExpandDims"hidden_layer_0_1/rnn/strided_sliceMhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims/dim*
T0*

Tdim0
r
Dhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ConstConst*
dtype0*
valueB:f
t
Jhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat/axisConst*
dtype0*
value	B : 
�
Ehidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concatConcatV2Ihidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDimsDhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ConstJhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat/axis*
N*
T0*

Tidx0
w
Jhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros/ConstConst*
dtype0*
valueB
 *    
�
Dhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zerosFillEhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concatJhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros/Const*
T0*

index_type0
y
Ohidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dimConst*
dtype0*
value	B : 
�
Khidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1
ExpandDims"hidden_layer_0_1/rnn/strided_sliceOhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1/dim*
T0*

Tdim0
t
Fhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/Const_1Const*
dtype0*
valueB:f
y
Ohidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dimConst*
dtype0*
value	B : 
�
Khidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2
ExpandDims"hidden_layer_0_1/rnn/strided_sliceOhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2/dim*
T0*

Tdim0
t
Fhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Const*
dtype0*
valueB:f
v
Lhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axisConst*
dtype0*
value	B : 
�
Ghidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat_1ConcatV2Khidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_2Fhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/Const_2Lhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat_1/axis*
N*
T0*

Tidx0
y
Lhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/ConstConst*
dtype0*
valueB
 *    
�
Fhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1FillGhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/concat_1Lhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1/Const*
T0*

index_type0
y
Ohidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dimConst*
dtype0*
value	B : 
�
Khidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3
ExpandDims"hidden_layer_0_1/rnn/strided_sliceOhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3/dim*
T0*

Tdim0
t
Fhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/Const_3Const*
dtype0*
valueB:f
d
hidden_layer_0_1/rnn/Shape_1Shape$hidden_layer_0_1/rnn/sequence_length*
T0*
out_type0
d
hidden_layer_0_1/rnn/stackPack"hidden_layer_0_1/rnn/strided_slice*
N*
T0*

axis 
�
hidden_layer_0_1/rnn/EqualEqualhidden_layer_0_1/rnn/Shape_1hidden_layer_0_1/rnn/stack*
T0*
incompatible_shape_error(
H
hidden_layer_0_1/rnn/ConstConst*
dtype0*
valueB: 
t
hidden_layer_0_1/rnn/AllAllhidden_layer_0_1/rnn/Equalhidden_layer_0_1/rnn/Const*

Tidx0*
	keep_dims( 
�
!hidden_layer_0_1/rnn/Assert/ConstConst*
dtype0*U
valueLBJ BDExpected shape for Tensor hidden_layer_0_1/rnn/sequence_length:0 is 
\
#hidden_layer_0_1/rnn/Assert/Const_1Const*
dtype0*!
valueB B but saw shape: 
�
)hidden_layer_0_1/rnn/Assert/Assert/data_0Const*
dtype0*U
valueLBJ BDExpected shape for Tensor hidden_layer_0_1/rnn/sequence_length:0 is 
b
)hidden_layer_0_1/rnn/Assert/Assert/data_2Const*
dtype0*!
valueB B but saw shape: 
�
"hidden_layer_0_1/rnn/Assert/AssertAsserthidden_layer_0_1/rnn/All)hidden_layer_0_1/rnn/Assert/Assert/data_0hidden_layer_0_1/rnn/stack)hidden_layer_0_1/rnn/Assert/Assert/data_2hidden_layer_0_1/rnn/Shape_1*
T
2*
	summarize
�
 hidden_layer_0_1/rnn/CheckSeqLenIdentity$hidden_layer_0_1/rnn/sequence_length#^hidden_layer_0_1/rnn/Assert/Assert*
T0
^
hidden_layer_0_1/rnn/Shape_2Shapehidden_layer_0_1/rnn/transpose*
T0*
out_type0
X
*hidden_layer_0_1/rnn/strided_slice_1/stackConst*
dtype0*
valueB: 
Z
,hidden_layer_0_1/rnn/strided_slice_1/stack_1Const*
dtype0*
valueB:
Z
,hidden_layer_0_1/rnn/strided_slice_1/stack_2Const*
dtype0*
valueB:
�
$hidden_layer_0_1/rnn/strided_slice_1StridedSlicehidden_layer_0_1/rnn/Shape_2*hidden_layer_0_1/rnn/strided_slice_1/stack,hidden_layer_0_1/rnn/strided_slice_1/stack_1,hidden_layer_0_1/rnn/strided_slice_1/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
^
hidden_layer_0_1/rnn/Shape_3Shapehidden_layer_0_1/rnn/transpose*
T0*
out_type0
X
*hidden_layer_0_1/rnn/strided_slice_2/stackConst*
dtype0*
valueB:
Z
,hidden_layer_0_1/rnn/strided_slice_2/stack_1Const*
dtype0*
valueB:
Z
,hidden_layer_0_1/rnn/strided_slice_2/stack_2Const*
dtype0*
valueB:
�
$hidden_layer_0_1/rnn/strided_slice_2StridedSlicehidden_layer_0_1/rnn/Shape_3*hidden_layer_0_1/rnn/strided_slice_2/stack,hidden_layer_0_1/rnn/strided_slice_2/stack_1,hidden_layer_0_1/rnn/strided_slice_2/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
M
#hidden_layer_0_1/rnn/ExpandDims/dimConst*
dtype0*
value	B : 
�
hidden_layer_0_1/rnn/ExpandDims
ExpandDims$hidden_layer_0_1/rnn/strided_slice_2#hidden_layer_0_1/rnn/ExpandDims/dim*
T0*

Tdim0
J
hidden_layer_0_1/rnn/Const_1Const*
dtype0*
valueB:f
L
"hidden_layer_0_1/rnn/concat_1/axisConst*
dtype0*
value	B : 
�
hidden_layer_0_1/rnn/concat_1ConcatV2hidden_layer_0_1/rnn/ExpandDimshidden_layer_0_1/rnn/Const_1"hidden_layer_0_1/rnn/concat_1/axis*
N*
T0*

Tidx0
M
 hidden_layer_0_1/rnn/zeros/ConstConst*
dtype0*
valueB
 *    
~
hidden_layer_0_1/rnn/zerosFillhidden_layer_0_1/rnn/concat_1 hidden_layer_0_1/rnn/zeros/Const*
T0*

index_type0
J
hidden_layer_0_1/rnn/Const_2Const*
dtype0*
valueB: 
�
hidden_layer_0_1/rnn/MinMin hidden_layer_0_1/rnn/CheckSeqLenhidden_layer_0_1/rnn/Const_2*
T0*

Tidx0*
	keep_dims( 
J
hidden_layer_0_1/rnn/Const_3Const*
dtype0*
valueB: 
�
hidden_layer_0_1/rnn/MaxMax hidden_layer_0_1/rnn/CheckSeqLenhidden_layer_0_1/rnn/Const_3*
T0*

Tidx0*
	keep_dims( 
C
hidden_layer_0_1/rnn/timeConst*
dtype0*
value	B : 
�
 hidden_layer_0_1/rnn/TensorArrayTensorArrayV3$hidden_layer_0_1/rnn/strided_slice_1*
clear_after_read(*
dtype0*
dynamic_size( *$
element_shape:���������f*
identical_element_shapes(*@
tensor_array_name+)hidden_layer_0_1/rnn/dynamic_rnn/output_0
�
"hidden_layer_0_1/rnn/TensorArray_1TensorArrayV3$hidden_layer_0_1/rnn/strided_slice_1*
clear_after_read(*
dtype0*
dynamic_size( *%
element_shape:����������*
identical_element_shapes(*?
tensor_array_name*(hidden_layer_0_1/rnn/dynamic_rnn/input_0
o
-hidden_layer_0_1/rnn/TensorArrayUnstack/ShapeShapehidden_layer_0_1/rnn/transpose*
T0*
out_type0
i
;hidden_layer_0_1/rnn/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: 
k
=hidden_layer_0_1/rnn/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:
k
=hidden_layer_0_1/rnn/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
valueB:
�
5hidden_layer_0_1/rnn/TensorArrayUnstack/strided_sliceStridedSlice-hidden_layer_0_1/rnn/TensorArrayUnstack/Shape;hidden_layer_0_1/rnn/TensorArrayUnstack/strided_slice/stack=hidden_layer_0_1/rnn/TensorArrayUnstack/strided_slice/stack_1=hidden_layer_0_1/rnn/TensorArrayUnstack/strided_slice/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
]
3hidden_layer_0_1/rnn/TensorArrayUnstack/range/startConst*
dtype0*
value	B : 
]
3hidden_layer_0_1/rnn/TensorArrayUnstack/range/deltaConst*
dtype0*
value	B :
�
-hidden_layer_0_1/rnn/TensorArrayUnstack/rangeRange3hidden_layer_0_1/rnn/TensorArrayUnstack/range/start5hidden_layer_0_1/rnn/TensorArrayUnstack/strided_slice3hidden_layer_0_1/rnn/TensorArrayUnstack/range/delta*

Tidx0
�
Ohidden_layer_0_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3"hidden_layer_0_1/rnn/TensorArray_1-hidden_layer_0_1/rnn/TensorArrayUnstack/rangehidden_layer_0_1/rnn/transpose$hidden_layer_0_1/rnn/TensorArray_1:1*
T0*1
_class'
%#loc:@hidden_layer_0_1/rnn/transpose
H
hidden_layer_0_1/rnn/Maximum/xConst*
dtype0*
value	B :
j
hidden_layer_0_1/rnn/MaximumMaximumhidden_layer_0_1/rnn/Maximum/xhidden_layer_0_1/rnn/Max*
T0
t
hidden_layer_0_1/rnn/MinimumMinimum$hidden_layer_0_1/rnn/strided_slice_1hidden_layer_0_1/rnn/Maximum*
T0
V
,hidden_layer_0_1/rnn/while/iteration_counterConst*
dtype0*
value	B : 
�
 hidden_layer_0_1/rnn/while/EnterEnter,hidden_layer_0_1/rnn/while/iteration_counter*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
"hidden_layer_0_1/rnn/while/Enter_1Enterhidden_layer_0_1/rnn/time*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
"hidden_layer_0_1/rnn/while/Enter_2Enter"hidden_layer_0_1/rnn/TensorArray:1*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
"hidden_layer_0_1/rnn/while/Enter_3EnterDhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
"hidden_layer_0_1/rnn/while/Enter_4EnterFhidden_layer_0_1/rnn/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
 hidden_layer_0_1/rnn/while/MergeMerge hidden_layer_0_1/rnn/while/Enter(hidden_layer_0_1/rnn/while/NextIteration*
N*
T0
�
"hidden_layer_0_1/rnn/while/Merge_1Merge"hidden_layer_0_1/rnn/while/Enter_1*hidden_layer_0_1/rnn/while/NextIteration_1*
N*
T0
�
"hidden_layer_0_1/rnn/while/Merge_2Merge"hidden_layer_0_1/rnn/while/Enter_2*hidden_layer_0_1/rnn/while/NextIteration_2*
N*
T0
�
"hidden_layer_0_1/rnn/while/Merge_3Merge"hidden_layer_0_1/rnn/while/Enter_3*hidden_layer_0_1/rnn/while/NextIteration_3*
N*
T0
�
"hidden_layer_0_1/rnn/while/Merge_4Merge"hidden_layer_0_1/rnn/while/Enter_4*hidden_layer_0_1/rnn/while/NextIteration_4*
N*
T0
y
hidden_layer_0_1/rnn/while/LessLess hidden_layer_0_1/rnn/while/Merge%hidden_layer_0_1/rnn/while/Less/Enter*
T0
�
%hidden_layer_0_1/rnn/while/Less/EnterEnter$hidden_layer_0_1/rnn/strided_slice_1*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 

!hidden_layer_0_1/rnn/while/Less_1Less"hidden_layer_0_1/rnn/while/Merge_1'hidden_layer_0_1/rnn/while/Less_1/Enter*
T0
�
'hidden_layer_0_1/rnn/while/Less_1/EnterEnterhidden_layer_0_1/rnn/Minimum*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
w
%hidden_layer_0_1/rnn/while/LogicalAnd
LogicalAndhidden_layer_0_1/rnn/while/Less!hidden_layer_0_1/rnn/while/Less_1
V
#hidden_layer_0_1/rnn/while/LoopCondLoopCond%hidden_layer_0_1/rnn/while/LogicalAnd
�
!hidden_layer_0_1/rnn/while/SwitchSwitch hidden_layer_0_1/rnn/while/Merge#hidden_layer_0_1/rnn/while/LoopCond*
T0*3
_class)
'%loc:@hidden_layer_0_1/rnn/while/Merge
�
#hidden_layer_0_1/rnn/while/Switch_1Switch"hidden_layer_0_1/rnn/while/Merge_1#hidden_layer_0_1/rnn/while/LoopCond*
T0*5
_class+
)'loc:@hidden_layer_0_1/rnn/while/Merge_1
�
#hidden_layer_0_1/rnn/while/Switch_2Switch"hidden_layer_0_1/rnn/while/Merge_2#hidden_layer_0_1/rnn/while/LoopCond*
T0*5
_class+
)'loc:@hidden_layer_0_1/rnn/while/Merge_2
�
#hidden_layer_0_1/rnn/while/Switch_3Switch"hidden_layer_0_1/rnn/while/Merge_3#hidden_layer_0_1/rnn/while/LoopCond*
T0*5
_class+
)'loc:@hidden_layer_0_1/rnn/while/Merge_3
�
#hidden_layer_0_1/rnn/while/Switch_4Switch"hidden_layer_0_1/rnn/while/Merge_4#hidden_layer_0_1/rnn/while/LoopCond*
T0*5
_class+
)'loc:@hidden_layer_0_1/rnn/while/Merge_4
]
#hidden_layer_0_1/rnn/while/IdentityIdentity#hidden_layer_0_1/rnn/while/Switch:1*
T0
a
%hidden_layer_0_1/rnn/while/Identity_1Identity%hidden_layer_0_1/rnn/while/Switch_1:1*
T0
a
%hidden_layer_0_1/rnn/while/Identity_2Identity%hidden_layer_0_1/rnn/while/Switch_2:1*
T0
a
%hidden_layer_0_1/rnn/while/Identity_3Identity%hidden_layer_0_1/rnn/while/Switch_3:1*
T0
a
%hidden_layer_0_1/rnn/while/Identity_4Identity%hidden_layer_0_1/rnn/while/Switch_4:1*
T0
p
 hidden_layer_0_1/rnn/while/add/yConst$^hidden_layer_0_1/rnn/while/Identity*
dtype0*
value	B :
w
hidden_layer_0_1/rnn/while/addAddV2#hidden_layer_0_1/rnn/while/Identity hidden_layer_0_1/rnn/while/add/y*
T0
�
,hidden_layer_0_1/rnn/while/TensorArrayReadV3TensorArrayReadV32hidden_layer_0_1/rnn/while/TensorArrayReadV3/Enter%hidden_layer_0_1/rnn/while/Identity_14hidden_layer_0_1/rnn/while/TensorArrayReadV3/Enter_1*
dtype0
�
2hidden_layer_0_1/rnn/while/TensorArrayReadV3/EnterEnter"hidden_layer_0_1/rnn/TensorArray_1*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
4hidden_layer_0_1/rnn/while/TensorArrayReadV3/Enter_1EnterOhidden_layer_0_1/rnn/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
'hidden_layer_0_1/rnn/while/GreaterEqualGreaterEqual%hidden_layer_0_1/rnn/while/Identity_1-hidden_layer_0_1/rnn/while/GreaterEqual/Enter*
T0
�
-hidden_layer_0_1/rnn/while/GreaterEqual/EnterEnter hidden_layer_0_1/rnn/CheckSeqLen*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
0hidden_layer_0_1/rnn/while/lstm_cell/concat/axisConst$^hidden_layer_0_1/rnn/while/Identity*
dtype0*
value	B :
�
+hidden_layer_0_1/rnn/while/lstm_cell/concatConcatV2,hidden_layer_0_1/rnn/while/TensorArrayReadV3%hidden_layer_0_1/rnn/while/Identity_40hidden_layer_0_1/rnn/while/lstm_cell/concat/axis*
N*
T0*

Tidx0
�
+hidden_layer_0_1/rnn/while/lstm_cell/MatMulMatMul+hidden_layer_0_1/rnn/while/lstm_cell/concat1hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter*
T0*
transpose_a( *
transpose_b( 
�
1hidden_layer_0_1/rnn/while/lstm_cell/MatMul/EnterEnter(hidden_layer_0/rnn/lstm_cell/kernel/read*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
,hidden_layer_0_1/rnn/while/lstm_cell/BiasAddBiasAdd+hidden_layer_0_1/rnn/while/lstm_cell/MatMul2hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter*
T0*
data_formatNHWC
�
2hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/EnterEnter&hidden_layer_0/rnn/lstm_cell/bias/read*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
4hidden_layer_0_1/rnn/while/lstm_cell/split/split_dimConst$^hidden_layer_0_1/rnn/while/Identity*
dtype0*
value	B :
�
*hidden_layer_0_1/rnn/while/lstm_cell/splitSplit4hidden_layer_0_1/rnn/while/lstm_cell/split/split_dim,hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd*
T0*
	num_split
}
*hidden_layer_0_1/rnn/while/lstm_cell/add/yConst$^hidden_layer_0_1/rnn/while/Identity*
dtype0*
valueB
 *  �?
�
(hidden_layer_0_1/rnn/while/lstm_cell/addAddV2,hidden_layer_0_1/rnn/while/lstm_cell/split:2*hidden_layer_0_1/rnn/while/lstm_cell/add/y*
T0
j
,hidden_layer_0_1/rnn/while/lstm_cell/SigmoidSigmoid(hidden_layer_0_1/rnn/while/lstm_cell/add*
T0
�
(hidden_layer_0_1/rnn/while/lstm_cell/mulMul,hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid%hidden_layer_0_1/rnn/while/Identity_3*
T0
n
.hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_1Sigmoid*hidden_layer_0_1/rnn/while/lstm_cell/split*
T0
h
)hidden_layer_0_1/rnn/while/lstm_cell/TanhTanh,hidden_layer_0_1/rnn/while/lstm_cell/split:1*
T0
�
*hidden_layer_0_1/rnn/while/lstm_cell/mul_1Mul.hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_1)hidden_layer_0_1/rnn/while/lstm_cell/Tanh*
T0
�
*hidden_layer_0_1/rnn/while/lstm_cell/add_1AddV2(hidden_layer_0_1/rnn/while/lstm_cell/mul*hidden_layer_0_1/rnn/while/lstm_cell/mul_1*
T0
p
.hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_2Sigmoid,hidden_layer_0_1/rnn/while/lstm_cell/split:3*
T0
h
+hidden_layer_0_1/rnn/while/lstm_cell/Tanh_1Tanh*hidden_layer_0_1/rnn/while/lstm_cell/add_1*
T0
�
*hidden_layer_0_1/rnn/while/lstm_cell/mul_2Mul.hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_2+hidden_layer_0_1/rnn/while/lstm_cell/Tanh_1*
T0
s
 hidden_layer_0_1/rnn/while/sub/xConst$^hidden_layer_0_1/rnn/while/Identity*
dtype0*
valueB
 *  �?
v
hidden_layer_0_1/rnn/while/subSub hidden_layer_0_1/rnn/while/sub/x$hidden_layer_0_1/rnn/while/sub/Enter*
T0
�
$hidden_layer_0_1/rnn/while/sub/EnterEnterkeep_prob_1*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
{
(hidden_layer_0_1/rnn/while/dropout/ConstConst$^hidden_layer_0_1/rnn/while/Identity*
dtype0*
valueB
 *  �?
�
&hidden_layer_0_1/rnn/while/dropout/SubSub(hidden_layer_0_1/rnn/while/dropout/Consthidden_layer_0_1/rnn/while/sub*
T0
�
*hidden_layer_0_1/rnn/while/dropout/RealDivRealDiv*hidden_layer_0_1/rnn/while/lstm_cell/mul_2&hidden_layer_0_1/rnn/while/dropout/Sub*
T0
v
(hidden_layer_0_1/rnn/while/dropout/ShapeShape*hidden_layer_0_1/rnn/while/lstm_cell/mul_2*
T0*
out_type0
�
?hidden_layer_0_1/rnn/while/dropout/random_uniform/RandomUniformRandomUniform(hidden_layer_0_1/rnn/while/dropout/Shape*
T0*
dtype0*

seed *
seed2 
�
/hidden_layer_0_1/rnn/while/dropout/GreaterEqualGreaterEqual?hidden_layer_0_1/rnn/while/dropout/random_uniform/RandomUniformhidden_layer_0_1/rnn/while/sub*
T0
�
'hidden_layer_0_1/rnn/while/dropout/CastCast/hidden_layer_0_1/rnn/while/dropout/GreaterEqual*

DstT0*

SrcT0
*
Truncate( 
�
&hidden_layer_0_1/rnn/while/dropout/MulMul*hidden_layer_0_1/rnn/while/dropout/RealDiv'hidden_layer_0_1/rnn/while/dropout/Cast*
T0
�
!hidden_layer_0_1/rnn/while/SelectSelect'hidden_layer_0_1/rnn/while/GreaterEqual'hidden_layer_0_1/rnn/while/Select/Enter&hidden_layer_0_1/rnn/while/dropout/Mul*
T0*9
_class/
-+loc:@hidden_layer_0_1/rnn/while/dropout/Mul
�
'hidden_layer_0_1/rnn/while/Select/EnterEnterhidden_layer_0_1/rnn/zeros*
T0*9
_class/
-+loc:@hidden_layer_0_1/rnn/while/dropout/Mul*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
#hidden_layer_0_1/rnn/while/Select_1Select'hidden_layer_0_1/rnn/while/GreaterEqual%hidden_layer_0_1/rnn/while/Identity_3*hidden_layer_0_1/rnn/while/lstm_cell/add_1*
T0*=
_class3
1/loc:@hidden_layer_0_1/rnn/while/lstm_cell/add_1
�
#hidden_layer_0_1/rnn/while/Select_2Select'hidden_layer_0_1/rnn/while/GreaterEqual%hidden_layer_0_1/rnn/while/Identity_4*hidden_layer_0_1/rnn/while/lstm_cell/mul_2*
T0*=
_class3
1/loc:@hidden_layer_0_1/rnn/while/lstm_cell/mul_2
�
>hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Dhidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/Enter%hidden_layer_0_1/rnn/while/Identity_1!hidden_layer_0_1/rnn/while/Select%hidden_layer_0_1/rnn/while/Identity_2*
T0*9
_class/
-+loc:@hidden_layer_0_1/rnn/while/dropout/Mul
�
Dhidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnter hidden_layer_0_1/rnn/TensorArray*
T0*9
_class/
-+loc:@hidden_layer_0_1/rnn/while/dropout/Mul*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
r
"hidden_layer_0_1/rnn/while/add_1/yConst$^hidden_layer_0_1/rnn/while/Identity*
dtype0*
value	B :
}
 hidden_layer_0_1/rnn/while/add_1AddV2%hidden_layer_0_1/rnn/while/Identity_1"hidden_layer_0_1/rnn/while/add_1/y*
T0
b
(hidden_layer_0_1/rnn/while/NextIterationNextIterationhidden_layer_0_1/rnn/while/add*
T0
f
*hidden_layer_0_1/rnn/while/NextIteration_1NextIteration hidden_layer_0_1/rnn/while/add_1*
T0
�
*hidden_layer_0_1/rnn/while/NextIteration_2NextIteration>hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3*
T0
i
*hidden_layer_0_1/rnn/while/NextIteration_3NextIteration#hidden_layer_0_1/rnn/while/Select_1*
T0
i
*hidden_layer_0_1/rnn/while/NextIteration_4NextIteration#hidden_layer_0_1/rnn/while/Select_2*
T0
S
hidden_layer_0_1/rnn/while/ExitExit!hidden_layer_0_1/rnn/while/Switch*
T0
W
!hidden_layer_0_1/rnn/while/Exit_1Exit#hidden_layer_0_1/rnn/while/Switch_1*
T0
W
!hidden_layer_0_1/rnn/while/Exit_2Exit#hidden_layer_0_1/rnn/while/Switch_2*
T0
W
!hidden_layer_0_1/rnn/while/Exit_3Exit#hidden_layer_0_1/rnn/while/Switch_3*
T0
W
!hidden_layer_0_1/rnn/while/Exit_4Exit#hidden_layer_0_1/rnn/while/Switch_4*
T0
�
1hidden_layer_0_1/rnn/TensorArrayStack/range/startConst*3
_class)
'%loc:@hidden_layer_0_1/rnn/TensorArray*
dtype0*
value	B : 
�
1hidden_layer_0_1/rnn/TensorArrayStack/range/deltaConst*3
_class)
'%loc:@hidden_layer_0_1/rnn/TensorArray*
dtype0*
value	B :
�
+hidden_layer_0_1/rnn/TensorArrayStack/rangeRange1hidden_layer_0_1/rnn/TensorArrayStack/range/start$hidden_layer_0_1/rnn/strided_slice_11hidden_layer_0_1/rnn/TensorArrayStack/range/delta*

Tidx0*3
_class)
'%loc:@hidden_layer_0_1/rnn/TensorArray
�
9hidden_layer_0_1/rnn/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3 hidden_layer_0_1/rnn/TensorArray+hidden_layer_0_1/rnn/TensorArrayStack/range!hidden_layer_0_1/rnn/while/Exit_2*3
_class)
'%loc:@hidden_layer_0_1/rnn/TensorArray*
dtype0*$
element_shape:���������f
J
hidden_layer_0_1/rnn/Const_4Const*
dtype0*
valueB:f
E
hidden_layer_0_1/rnn/Rank_1Const*
dtype0*
value	B :
L
"hidden_layer_0_1/rnn/range_1/startConst*
dtype0*
value	B :
L
"hidden_layer_0_1/rnn/range_1/deltaConst*
dtype0*
value	B :
�
hidden_layer_0_1/rnn/range_1Range"hidden_layer_0_1/rnn/range_1/starthidden_layer_0_1/rnn/Rank_1"hidden_layer_0_1/rnn/range_1/delta*

Tidx0
[
&hidden_layer_0_1/rnn/concat_2/values_0Const*
dtype0*
valueB"       
L
"hidden_layer_0_1/rnn/concat_2/axisConst*
dtype0*
value	B : 
�
hidden_layer_0_1/rnn/concat_2ConcatV2&hidden_layer_0_1/rnn/concat_2/values_0hidden_layer_0_1/rnn/range_1"hidden_layer_0_1/rnn/concat_2/axis*
N*
T0*

Tidx0
�
 hidden_layer_0_1/rnn/transpose_1	Transpose9hidden_layer_0_1/rnn/TensorArrayStack/TensorArrayGatherV3hidden_layer_0_1/rnn/concat_2*
T0*
Tperm0
X
output_layer_1/ShapeShape hidden_layer_0_1/rnn/transpose_1*
T0*
out_type0
P
"output_layer_1/strided_slice/stackConst*
dtype0*
valueB:
R
$output_layer_1/strided_slice/stack_1Const*
dtype0*
valueB:
R
$output_layer_1/strided_slice/stack_2Const*
dtype0*
valueB:
�
output_layer_1/strided_sliceStridedSliceoutput_layer_1/Shape"output_layer_1/strided_slice/stack$output_layer_1/strided_slice/stack_1$output_layer_1/strided_slice/stack_2*
Index0*
T0*

begin_mask *
ellipsis_mask *
end_mask *
new_axis_mask *
shrink_axis_mask
Q
output_layer_1/Reshape/shapeConst*
dtype0*
valueB"����f   
x
output_layer_1/ReshapeReshape hidden_layer_0_1/rnn/transpose_1output_layer_1/Reshape/shape*
T0*
Tshape0
�
output_layer_1/MatMulMatMuloutput_layer_1/Reshapeoutput_layer/weights/read*
T0*
transpose_a( *
transpose_b( 
U
output_layer_1/addAddV2output_layer_1/MatMuloutput_layer/biases/read*
T0
S
 output_layer_1/Reshape_1/shape/0Const*
dtype0*
valueB :
���������
K
 output_layer_1/Reshape_1/shape/2Const*
dtype0*
value
B :�
�
output_layer_1/Reshape_1/shapePack output_layer_1/Reshape_1/shape/0output_layer_1/strided_slice output_layer_1/Reshape_1/shape/2*
N*
T0*

axis 
n
output_layer_1/Reshape_1Reshapeoutput_layer_1/addoutput_layer_1/Reshape_1/shape*
T0*
Tshape0
B
output_layer_1/predsSigmoidoutput_layer_1/Reshape_1*
T0
F
output_layer_1/NotEqual/yConst*
dtype0*
valueB
 *    
p
output_layer_1/NotEqualNotEqualy_seq_1output_layer_1/NotEqual/y*
T0*
incompatible_shape_error(
?
output_layer_1/WhereWhereoutput_layer_1/NotEqual*
T0

r
output_layer_1/GatherNdGatherNdoutput_layer_1/Reshape_1output_layer_1/Where*
Tindices0	*
Tparams0
p
output_layer_1/GatherNd_1GatherNdoutput_layer_1/predsoutput_layer_1/Where*
Tindices0	*
Tparams0
d
output_layer_1/GatherNd_2GatherNdy_corr_1output_layer_1/Where*
Tindices0	*
Tparams0
V
'output_layer_1/logistic_loss/zeros_like	ZerosLikeoutput_layer_1/GatherNd*
T0
�
)output_layer_1/logistic_loss/GreaterEqualGreaterEqualoutput_layer_1/GatherNd'output_layer_1/logistic_loss/zeros_like*
T0
�
#output_layer_1/logistic_loss/SelectSelect)output_layer_1/logistic_loss/GreaterEqualoutput_layer_1/GatherNd'output_layer_1/logistic_loss/zeros_like*
T0
I
 output_layer_1/logistic_loss/NegNegoutput_layer_1/GatherNd*
T0
�
%output_layer_1/logistic_loss/Select_1Select)output_layer_1/logistic_loss/GreaterEqual output_layer_1/logistic_loss/Negoutput_layer_1/GatherNd*
T0
d
 output_layer_1/logistic_loss/mulMuloutput_layer_1/GatherNdoutput_layer_1/GatherNd_2*
T0
w
 output_layer_1/logistic_loss/subSub#output_layer_1/logistic_loss/Select output_layer_1/logistic_loss/mul*
T0
W
 output_layer_1/logistic_loss/ExpExp%output_layer_1/logistic_loss/Select_1*
T0
V
"output_layer_1/logistic_loss/Log1pLog1p output_layer_1/logistic_loss/Exp*
T0
t
output_layer_1/logistic_lossAddV2 output_layer_1/logistic_loss/sub"output_layer_1/logistic_loss/Log1p*
T0
B
output_layer_1/ConstConst*
dtype0*
valueB: 
u
output_layer_1/MeanMeanoutput_layer_1/logistic_lossoutput_layer_1/Const*
T0*

Tidx0*
	keep_dims( 
]
$output_layer_1/strided_slice_1/stackConst*
dtype0*!
valueB"            
_
&output_layer_1/strided_slice_1/stack_1Const*
dtype0*!
valueB"        I  
_
&output_layer_1/strided_slice_1/stack_2Const*
dtype0*!
valueB"         
�
output_layer_1/strided_slice_1StridedSliceX_1$output_layer_1/strided_slice_1/stack&output_layer_1/strided_slice_1/stack_1&output_layer_1/strided_slice_1/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
]
$output_layer_1/strided_slice_2/stackConst*
dtype0*!
valueB"        I  
_
&output_layer_1/strided_slice_2/stack_1Const*
dtype0*!
valueB"            
_
&output_layer_1/strided_slice_2/stack_2Const*
dtype0*!
valueB"         
�
output_layer_1/strided_slice_2StridedSliceX_1$output_layer_1/strided_slice_2/stack&output_layer_1/strided_slice_2/stack_1&output_layer_1/strided_slice_2/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
H
output_layer_1/NotEqual_1/yConst*
dtype0*
valueB
 *    
�
output_layer_1/NotEqual_1NotEqualoutput_layer_1/strided_slice_1output_layer_1/NotEqual_1/y*
T0*
incompatible_shape_error(
C
output_layer_1/Where_1Whereoutput_layer_1/NotEqual_1*
T0

v
output_layer_1/GatherNd_3GatherNdoutput_layer_1/Reshape_1output_layer_1/Where_1*
Tindices0	*
Tparams0
r
output_layer_1/GatherNd_4GatherNdoutput_layer_1/predsoutput_layer_1/Where_1*
Tindices0	*
Tparams0
|
output_layer_1/GatherNd_5GatherNdoutput_layer_1/strided_slice_2output_layer_1/Where_1*
Tindices0	*
Tparams0
Z
)output_layer_1/logistic_loss_1/zeros_like	ZerosLikeoutput_layer_1/GatherNd_3*
T0
�
+output_layer_1/logistic_loss_1/GreaterEqualGreaterEqualoutput_layer_1/GatherNd_3)output_layer_1/logistic_loss_1/zeros_like*
T0
�
%output_layer_1/logistic_loss_1/SelectSelect+output_layer_1/logistic_loss_1/GreaterEqualoutput_layer_1/GatherNd_3)output_layer_1/logistic_loss_1/zeros_like*
T0
M
"output_layer_1/logistic_loss_1/NegNegoutput_layer_1/GatherNd_3*
T0
�
'output_layer_1/logistic_loss_1/Select_1Select+output_layer_1/logistic_loss_1/GreaterEqual"output_layer_1/logistic_loss_1/Negoutput_layer_1/GatherNd_3*
T0
h
"output_layer_1/logistic_loss_1/mulMuloutput_layer_1/GatherNd_3output_layer_1/GatherNd_5*
T0
}
"output_layer_1/logistic_loss_1/subSub%output_layer_1/logistic_loss_1/Select"output_layer_1/logistic_loss_1/mul*
T0
[
"output_layer_1/logistic_loss_1/ExpExp'output_layer_1/logistic_loss_1/Select_1*
T0
Z
$output_layer_1/logistic_loss_1/Log1pLog1p"output_layer_1/logistic_loss_1/Exp*
T0
z
output_layer_1/logistic_loss_1AddV2"output_layer_1/logistic_loss_1/sub$output_layer_1/logistic_loss_1/Log1p*
T0
D
output_layer_1/Const_1Const*
dtype0*
valueB: 
{
output_layer_1/Mean_1Meanoutput_layer_1/logistic_loss_1output_layer_1/Const_1*
T0*

Tidx0*
	keep_dims( 
A
output_layer_1/mul/xConst*
dtype0*
valueB
 *���=
O
output_layer_1/mulMuloutput_layer_1/mul/xoutput_layer_1/Mean_1*
T0
O
output_layer_1/add_1AddV2output_layer_1/Meanoutput_layer_1/mul*
T0
+
output_layer_1/AbsAbsy_seq_1*
T0
N
$output_layer_1/Max/reduction_indicesConst*
dtype0*
value	B :
y
output_layer_1/MaxMaxoutput_layer_1/Abs$output_layer_1/Max/reduction_indices*
T0*

Tidx0*
	keep_dims( 
8
output_layer_1/SignSignoutput_layer_1/Max*
T0
N
$output_layer_1/Sum/reduction_indicesConst*
dtype0*
value	B :
z
output_layer_1/SumSumoutput_layer_1/Sign$output_layer_1/Sum/reduction_indices*
T0*

Tidx0*
	keep_dims( 
W
output_layer_1/CastCastoutput_layer_1/Sum*

DstT0*

SrcT0*
Truncate( 
Z
output_layer_1/Cast_1Castoutput_layer_1/Cast*

DstT0*

SrcT0*
Truncate( 
D
output_layer_1/Const_2Const*
dtype0*
valueB: 
p
output_layer_1/Sum_1Sumoutput_layer_1/Cast_1output_layer_1/Const_2*
T0*

Tidx0*
	keep_dims( 
]
$output_layer_1/strided_slice_3/stackConst*
dtype0*!
valueB"           
_
&output_layer_1/strided_slice_3/stack_1Const*
dtype0*!
valueB"            
_
&output_layer_1/strided_slice_3/stack_2Const*
dtype0*!
valueB"         
�
output_layer_1/strided_slice_3StridedSliceoutput_layer_1/preds$output_layer_1/strided_slice_3/stack&output_layer_1/strided_slice_3/stack_1&output_layer_1/strided_slice_3/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
]
$output_layer_1/strided_slice_4/stackConst*
dtype0*!
valueB"            
_
&output_layer_1/strided_slice_4/stack_1Const*
dtype0*!
valueB"    ����    
_
&output_layer_1/strided_slice_4/stack_2Const*
dtype0*!
valueB"         
�
output_layer_1/strided_slice_4StridedSliceoutput_layer_1/preds$output_layer_1/strided_slice_4/stack&output_layer_1/strided_slice_4/stack_1&output_layer_1/strided_slice_4/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
b
output_layer_1/subSuboutput_layer_1/strided_slice_3output_layer_1/strided_slice_4*
T0
8
output_layer_1/Abs_1Absoutput_layer_1/sub*
T0
O
output_layer_1/Const_3Const*
dtype0*!
valueB"          
o
output_layer_1/Sum_2Sumoutput_layer_1/Abs_1output_layer_1/Const_3*
T0*

Tidx0*
	keep_dims( 
V
output_layer_1/truedivRealDivoutput_layer_1/Sum_2output_layer_1/Sum_1*
T0
G
output_layer_1/truediv_1/yConst*
dtype0*
valueB
 *  �D
`
output_layer_1/truediv_1RealDivoutput_layer_1/truedivoutput_layer_1/truediv_1/y*
T0
C
output_layer_1/mul_1/xConst*
dtype0*
valueB
 *���<
V
output_layer_1/mul_1Muloutput_layer_1/mul_1/xoutput_layer_1/truediv_1*
T0
R
output_layer_1/add_2AddV2output_layer_1/add_1output_layer_1/mul_1*
T0
]
$output_layer_1/strided_slice_5/stackConst*
dtype0*!
valueB"           
_
&output_layer_1/strided_slice_5/stack_1Const*
dtype0*!
valueB"            
_
&output_layer_1/strided_slice_5/stack_2Const*
dtype0*!
valueB"         
�
output_layer_1/strided_slice_5StridedSliceoutput_layer_1/preds$output_layer_1/strided_slice_5/stack&output_layer_1/strided_slice_5/stack_1&output_layer_1/strided_slice_5/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
]
$output_layer_1/strided_slice_6/stackConst*
dtype0*!
valueB"            
_
&output_layer_1/strided_slice_6/stack_1Const*
dtype0*!
valueB"    ����    
_
&output_layer_1/strided_slice_6/stack_2Const*
dtype0*!
valueB"         
�
output_layer_1/strided_slice_6StridedSliceoutput_layer_1/preds$output_layer_1/strided_slice_6/stack&output_layer_1/strided_slice_6/stack_1&output_layer_1/strided_slice_6/stack_2*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
d
output_layer_1/sub_1Suboutput_layer_1/strided_slice_5output_layer_1/strided_slice_6*
T0
>
output_layer_1/SquareSquareoutput_layer_1/sub_1*
T0
O
output_layer_1/Const_4Const*
dtype0*!
valueB"          
p
output_layer_1/Sum_3Sumoutput_layer_1/Squareoutput_layer_1/Const_4*
T0*

Tidx0*
	keep_dims( 
X
output_layer_1/truediv_2RealDivoutput_layer_1/Sum_3output_layer_1/Sum_1*
T0
G
output_layer_1/truediv_3/yConst*
dtype0*
valueB
 *  �D
b
output_layer_1/truediv_3RealDivoutput_layer_1/truediv_2output_layer_1/truediv_3/y*
T0
C
output_layer_1/mul_2/xConst*
dtype0*
valueB
 *  @@
V
output_layer_1/mul_2Muloutput_layer_1/mul_2/xoutput_layer_1/truediv_3*
T0
R
output_layer_1/add_3AddV2output_layer_1/add_2output_layer_1/mul_2*
T0
D
Optimizer_1/gradients/ShapeConst*
dtype0*
valueB 
R
%Optimizer_1/gradients/grad_ys_0/ConstConst*
dtype0*
valueB
 *  �?
�
Optimizer_1/gradients/grad_ys_0FillOptimizer_1/gradients/Shape%Optimizer_1/gradients/grad_ys_0/Const*
T0*

index_type0
G
Optimizer_1/gradients/f_countConst*
dtype0*
value	B : 
�
Optimizer_1/gradients/f_count_1EnterOptimizer_1/gradients/f_count*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
|
Optimizer_1/gradients/MergeMergeOptimizer_1/gradients/f_count_1#Optimizer_1/gradients/NextIteration*
N*
T0
q
Optimizer_1/gradients/SwitchSwitchOptimizer_1/gradients/Merge#hidden_layer_0_1/rnn/while/LoopCond*
T0
k
Optimizer_1/gradients/Add/yConst$^hidden_layer_0_1/rnn/while/Identity*
dtype0*
value	B :
h
Optimizer_1/gradients/AddAddV2Optimizer_1/gradients/Switch:1Optimizer_1/gradients/Add/y*
T0
�
#Optimizer_1/gradients/NextIterationNextIterationOptimizer_1/gradients/AddR^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPushV2V^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/StackPushV2V^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/StackPushV2x^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2o^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPushV2d^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPushV2f^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPushV2_1R^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/StackPushV2T^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/StackPushV2h^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPushV2V^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/StackPushV2Z^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/StackPushV2\^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2h^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2j^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1f^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2h^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2_1Z^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2h^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2j^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1V^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2X^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2h^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2j^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1V^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2X^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2f^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2h^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1V^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2*
T0
N
Optimizer_1/gradients/f_count_2ExitOptimizer_1/gradients/Switch*
T0
G
Optimizer_1/gradients/b_countConst*
dtype0*
value	B :
�
Optimizer_1/gradients/b_count_1EnterOptimizer_1/gradients/f_count_2*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
Optimizer_1/gradients/Merge_1MergeOptimizer_1/gradients/b_count_1%Optimizer_1/gradients/NextIteration_1*
N*
T0
�
"Optimizer_1/gradients/GreaterEqualGreaterEqualOptimizer_1/gradients/Merge_1(Optimizer_1/gradients/GreaterEqual/Enter*
T0
�
(Optimizer_1/gradients/GreaterEqual/EnterEnterOptimizer_1/gradients/b_count*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
O
Optimizer_1/gradients/b_count_2LoopCond"Optimizer_1/gradients/GreaterEqual
q
Optimizer_1/gradients/Switch_1SwitchOptimizer_1/gradients/Merge_1Optimizer_1/gradients/b_count_2*
T0
u
Optimizer_1/gradients/SubSub Optimizer_1/gradients/Switch_1:1(Optimizer_1/gradients/GreaterEqual/Enter*
T0
�
%Optimizer_1/gradients/NextIteration_1NextIterationOptimizer_1/gradients/Subj^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/b_sync*
T0
P
Optimizer_1/gradients/b_count_3ExitOptimizer_1/gradients/Switch_1*
T0
j
@Optimizer_1/gradients/output_layer_1/add_3_grad/tuple/group_depsNoOp ^Optimizer_1/gradients/grad_ys_0
�
HOptimizer_1/gradients/output_layer_1/add_3_grad/tuple/control_dependencyIdentityOptimizer_1/gradients/grad_ys_0A^Optimizer_1/gradients/output_layer_1/add_3_grad/tuple/group_deps*
T0*2
_class(
&$loc:@Optimizer_1/gradients/grad_ys_0
�
JOptimizer_1/gradients/output_layer_1/add_3_grad/tuple/control_dependency_1IdentityOptimizer_1/gradients/grad_ys_0A^Optimizer_1/gradients/output_layer_1/add_3_grad/tuple/group_deps*
T0*2
_class(
&$loc:@Optimizer_1/gradients/grad_ys_0
�
@Optimizer_1/gradients/output_layer_1/add_2_grad/tuple/group_depsNoOpI^Optimizer_1/gradients/output_layer_1/add_3_grad/tuple/control_dependency
�
HOptimizer_1/gradients/output_layer_1/add_2_grad/tuple/control_dependencyIdentityHOptimizer_1/gradients/output_layer_1/add_3_grad/tuple/control_dependencyA^Optimizer_1/gradients/output_layer_1/add_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@Optimizer_1/gradients/grad_ys_0
�
JOptimizer_1/gradients/output_layer_1/add_2_grad/tuple/control_dependency_1IdentityHOptimizer_1/gradients/output_layer_1/add_3_grad/tuple/control_dependencyA^Optimizer_1/gradients/output_layer_1/add_2_grad/tuple/group_deps*
T0*2
_class(
&$loc:@Optimizer_1/gradients/grad_ys_0
�
3Optimizer_1/gradients/output_layer_1/mul_2_grad/MulMulJOptimizer_1/gradients/output_layer_1/add_3_grad/tuple/control_dependency_1output_layer_1/truediv_3*
T0
�
5Optimizer_1/gradients/output_layer_1/mul_2_grad/Mul_1MulJOptimizer_1/gradients/output_layer_1/add_3_grad/tuple/control_dependency_1output_layer_1/mul_2/x*
T0
�
@Optimizer_1/gradients/output_layer_1/mul_2_grad/tuple/group_depsNoOp4^Optimizer_1/gradients/output_layer_1/mul_2_grad/Mul6^Optimizer_1/gradients/output_layer_1/mul_2_grad/Mul_1
�
HOptimizer_1/gradients/output_layer_1/mul_2_grad/tuple/control_dependencyIdentity3Optimizer_1/gradients/output_layer_1/mul_2_grad/MulA^Optimizer_1/gradients/output_layer_1/mul_2_grad/tuple/group_deps*
T0*F
_class<
:8loc:@Optimizer_1/gradients/output_layer_1/mul_2_grad/Mul
�
JOptimizer_1/gradients/output_layer_1/mul_2_grad/tuple/control_dependency_1Identity5Optimizer_1/gradients/output_layer_1/mul_2_grad/Mul_1A^Optimizer_1/gradients/output_layer_1/mul_2_grad/tuple/group_deps*
T0*H
_class>
<:loc:@Optimizer_1/gradients/output_layer_1/mul_2_grad/Mul_1
�
@Optimizer_1/gradients/output_layer_1/add_1_grad/tuple/group_depsNoOpI^Optimizer_1/gradients/output_layer_1/add_2_grad/tuple/control_dependency
�
HOptimizer_1/gradients/output_layer_1/add_1_grad/tuple/control_dependencyIdentityHOptimizer_1/gradients/output_layer_1/add_2_grad/tuple/control_dependencyA^Optimizer_1/gradients/output_layer_1/add_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@Optimizer_1/gradients/grad_ys_0
�
JOptimizer_1/gradients/output_layer_1/add_1_grad/tuple/control_dependency_1IdentityHOptimizer_1/gradients/output_layer_1/add_2_grad/tuple/control_dependencyA^Optimizer_1/gradients/output_layer_1/add_1_grad/tuple/group_deps*
T0*2
_class(
&$loc:@Optimizer_1/gradients/grad_ys_0
�
3Optimizer_1/gradients/output_layer_1/mul_1_grad/MulMulJOptimizer_1/gradients/output_layer_1/add_2_grad/tuple/control_dependency_1output_layer_1/truediv_1*
T0
�
5Optimizer_1/gradients/output_layer_1/mul_1_grad/Mul_1MulJOptimizer_1/gradients/output_layer_1/add_2_grad/tuple/control_dependency_1output_layer_1/mul_1/x*
T0
�
@Optimizer_1/gradients/output_layer_1/mul_1_grad/tuple/group_depsNoOp4^Optimizer_1/gradients/output_layer_1/mul_1_grad/Mul6^Optimizer_1/gradients/output_layer_1/mul_1_grad/Mul_1
�
HOptimizer_1/gradients/output_layer_1/mul_1_grad/tuple/control_dependencyIdentity3Optimizer_1/gradients/output_layer_1/mul_1_grad/MulA^Optimizer_1/gradients/output_layer_1/mul_1_grad/tuple/group_deps*
T0*F
_class<
:8loc:@Optimizer_1/gradients/output_layer_1/mul_1_grad/Mul
�
JOptimizer_1/gradients/output_layer_1/mul_1_grad/tuple/control_dependency_1Identity5Optimizer_1/gradients/output_layer_1/mul_1_grad/Mul_1A^Optimizer_1/gradients/output_layer_1/mul_1_grad/tuple/group_deps*
T0*H
_class>
<:loc:@Optimizer_1/gradients/output_layer_1/mul_1_grad/Mul_1
b
9Optimizer_1/gradients/output_layer_1/truediv_3_grad/ShapeConst*
dtype0*
valueB 
d
;Optimizer_1/gradients/output_layer_1/truediv_3_grad/Shape_1Const*
dtype0*
valueB 
�
IOptimizer_1/gradients/output_layer_1/truediv_3_grad/BroadcastGradientArgsBroadcastGradientArgs9Optimizer_1/gradients/output_layer_1/truediv_3_grad/Shape;Optimizer_1/gradients/output_layer_1/truediv_3_grad/Shape_1*
T0
�
;Optimizer_1/gradients/output_layer_1/truediv_3_grad/RealDivRealDivJOptimizer_1/gradients/output_layer_1/mul_2_grad/tuple/control_dependency_1output_layer_1/truediv_3/y*
T0
�
7Optimizer_1/gradients/output_layer_1/truediv_3_grad/SumSum;Optimizer_1/gradients/output_layer_1/truediv_3_grad/RealDivIOptimizer_1/gradients/output_layer_1/truediv_3_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
;Optimizer_1/gradients/output_layer_1/truediv_3_grad/ReshapeReshape7Optimizer_1/gradients/output_layer_1/truediv_3_grad/Sum9Optimizer_1/gradients/output_layer_1/truediv_3_grad/Shape*
T0*
Tshape0
a
7Optimizer_1/gradients/output_layer_1/truediv_3_grad/NegNegoutput_layer_1/truediv_2*
T0
�
=Optimizer_1/gradients/output_layer_1/truediv_3_grad/RealDiv_1RealDiv7Optimizer_1/gradients/output_layer_1/truediv_3_grad/Negoutput_layer_1/truediv_3/y*
T0
�
=Optimizer_1/gradients/output_layer_1/truediv_3_grad/RealDiv_2RealDiv=Optimizer_1/gradients/output_layer_1/truediv_3_grad/RealDiv_1output_layer_1/truediv_3/y*
T0
�
7Optimizer_1/gradients/output_layer_1/truediv_3_grad/mulMulJOptimizer_1/gradients/output_layer_1/mul_2_grad/tuple/control_dependency_1=Optimizer_1/gradients/output_layer_1/truediv_3_grad/RealDiv_2*
T0
�
9Optimizer_1/gradients/output_layer_1/truediv_3_grad/Sum_1Sum7Optimizer_1/gradients/output_layer_1/truediv_3_grad/mulKOptimizer_1/gradients/output_layer_1/truediv_3_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
=Optimizer_1/gradients/output_layer_1/truediv_3_grad/Reshape_1Reshape9Optimizer_1/gradients/output_layer_1/truediv_3_grad/Sum_1;Optimizer_1/gradients/output_layer_1/truediv_3_grad/Shape_1*
T0*
Tshape0
�
DOptimizer_1/gradients/output_layer_1/truediv_3_grad/tuple/group_depsNoOp<^Optimizer_1/gradients/output_layer_1/truediv_3_grad/Reshape>^Optimizer_1/gradients/output_layer_1/truediv_3_grad/Reshape_1
�
LOptimizer_1/gradients/output_layer_1/truediv_3_grad/tuple/control_dependencyIdentity;Optimizer_1/gradients/output_layer_1/truediv_3_grad/ReshapeE^Optimizer_1/gradients/output_layer_1/truediv_3_grad/tuple/group_deps*
T0*N
_classD
B@loc:@Optimizer_1/gradients/output_layer_1/truediv_3_grad/Reshape
�
NOptimizer_1/gradients/output_layer_1/truediv_3_grad/tuple/control_dependency_1Identity=Optimizer_1/gradients/output_layer_1/truediv_3_grad/Reshape_1E^Optimizer_1/gradients/output_layer_1/truediv_3_grad/tuple/group_deps*
T0*P
_classF
DBloc:@Optimizer_1/gradients/output_layer_1/truediv_3_grad/Reshape_1
j
<Optimizer_1/gradients/output_layer_1/Mean_grad/Reshape/shapeConst*
dtype0*
valueB:
�
6Optimizer_1/gradients/output_layer_1/Mean_grad/ReshapeReshapeHOptimizer_1/gradients/output_layer_1/add_1_grad/tuple/control_dependency<Optimizer_1/gradients/output_layer_1/Mean_grad/Reshape/shape*
T0*
Tshape0
t
4Optimizer_1/gradients/output_layer_1/Mean_grad/ShapeShapeoutput_layer_1/logistic_loss*
T0*
out_type0
�
3Optimizer_1/gradients/output_layer_1/Mean_grad/TileTile6Optimizer_1/gradients/output_layer_1/Mean_grad/Reshape4Optimizer_1/gradients/output_layer_1/Mean_grad/Shape*
T0*

Tmultiples0
v
6Optimizer_1/gradients/output_layer_1/Mean_grad/Shape_1Shapeoutput_layer_1/logistic_loss*
T0*
out_type0
_
6Optimizer_1/gradients/output_layer_1/Mean_grad/Shape_2Const*
dtype0*
valueB 
b
4Optimizer_1/gradients/output_layer_1/Mean_grad/ConstConst*
dtype0*
valueB: 
�
3Optimizer_1/gradients/output_layer_1/Mean_grad/ProdProd6Optimizer_1/gradients/output_layer_1/Mean_grad/Shape_14Optimizer_1/gradients/output_layer_1/Mean_grad/Const*
T0*

Tidx0*
	keep_dims( 
d
6Optimizer_1/gradients/output_layer_1/Mean_grad/Const_1Const*
dtype0*
valueB: 
�
5Optimizer_1/gradients/output_layer_1/Mean_grad/Prod_1Prod6Optimizer_1/gradients/output_layer_1/Mean_grad/Shape_26Optimizer_1/gradients/output_layer_1/Mean_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
b
8Optimizer_1/gradients/output_layer_1/Mean_grad/Maximum/yConst*
dtype0*
value	B :
�
6Optimizer_1/gradients/output_layer_1/Mean_grad/MaximumMaximum5Optimizer_1/gradients/output_layer_1/Mean_grad/Prod_18Optimizer_1/gradients/output_layer_1/Mean_grad/Maximum/y*
T0
�
7Optimizer_1/gradients/output_layer_1/Mean_grad/floordivFloorDiv3Optimizer_1/gradients/output_layer_1/Mean_grad/Prod6Optimizer_1/gradients/output_layer_1/Mean_grad/Maximum*
T0
�
3Optimizer_1/gradients/output_layer_1/Mean_grad/CastCast7Optimizer_1/gradients/output_layer_1/Mean_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
�
6Optimizer_1/gradients/output_layer_1/Mean_grad/truedivRealDiv3Optimizer_1/gradients/output_layer_1/Mean_grad/Tile3Optimizer_1/gradients/output_layer_1/Mean_grad/Cast*
T0
�
1Optimizer_1/gradients/output_layer_1/mul_grad/MulMulJOptimizer_1/gradients/output_layer_1/add_1_grad/tuple/control_dependency_1output_layer_1/Mean_1*
T0
�
3Optimizer_1/gradients/output_layer_1/mul_grad/Mul_1MulJOptimizer_1/gradients/output_layer_1/add_1_grad/tuple/control_dependency_1output_layer_1/mul/x*
T0
�
>Optimizer_1/gradients/output_layer_1/mul_grad/tuple/group_depsNoOp2^Optimizer_1/gradients/output_layer_1/mul_grad/Mul4^Optimizer_1/gradients/output_layer_1/mul_grad/Mul_1
�
FOptimizer_1/gradients/output_layer_1/mul_grad/tuple/control_dependencyIdentity1Optimizer_1/gradients/output_layer_1/mul_grad/Mul?^Optimizer_1/gradients/output_layer_1/mul_grad/tuple/group_deps*
T0*D
_class:
86loc:@Optimizer_1/gradients/output_layer_1/mul_grad/Mul
�
HOptimizer_1/gradients/output_layer_1/mul_grad/tuple/control_dependency_1Identity3Optimizer_1/gradients/output_layer_1/mul_grad/Mul_1?^Optimizer_1/gradients/output_layer_1/mul_grad/tuple/group_deps*
T0*F
_class<
:8loc:@Optimizer_1/gradients/output_layer_1/mul_grad/Mul_1
b
9Optimizer_1/gradients/output_layer_1/truediv_1_grad/ShapeConst*
dtype0*
valueB 
d
;Optimizer_1/gradients/output_layer_1/truediv_1_grad/Shape_1Const*
dtype0*
valueB 
�
IOptimizer_1/gradients/output_layer_1/truediv_1_grad/BroadcastGradientArgsBroadcastGradientArgs9Optimizer_1/gradients/output_layer_1/truediv_1_grad/Shape;Optimizer_1/gradients/output_layer_1/truediv_1_grad/Shape_1*
T0
�
;Optimizer_1/gradients/output_layer_1/truediv_1_grad/RealDivRealDivJOptimizer_1/gradients/output_layer_1/mul_1_grad/tuple/control_dependency_1output_layer_1/truediv_1/y*
T0
�
7Optimizer_1/gradients/output_layer_1/truediv_1_grad/SumSum;Optimizer_1/gradients/output_layer_1/truediv_1_grad/RealDivIOptimizer_1/gradients/output_layer_1/truediv_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
;Optimizer_1/gradients/output_layer_1/truediv_1_grad/ReshapeReshape7Optimizer_1/gradients/output_layer_1/truediv_1_grad/Sum9Optimizer_1/gradients/output_layer_1/truediv_1_grad/Shape*
T0*
Tshape0
_
7Optimizer_1/gradients/output_layer_1/truediv_1_grad/NegNegoutput_layer_1/truediv*
T0
�
=Optimizer_1/gradients/output_layer_1/truediv_1_grad/RealDiv_1RealDiv7Optimizer_1/gradients/output_layer_1/truediv_1_grad/Negoutput_layer_1/truediv_1/y*
T0
�
=Optimizer_1/gradients/output_layer_1/truediv_1_grad/RealDiv_2RealDiv=Optimizer_1/gradients/output_layer_1/truediv_1_grad/RealDiv_1output_layer_1/truediv_1/y*
T0
�
7Optimizer_1/gradients/output_layer_1/truediv_1_grad/mulMulJOptimizer_1/gradients/output_layer_1/mul_1_grad/tuple/control_dependency_1=Optimizer_1/gradients/output_layer_1/truediv_1_grad/RealDiv_2*
T0
�
9Optimizer_1/gradients/output_layer_1/truediv_1_grad/Sum_1Sum7Optimizer_1/gradients/output_layer_1/truediv_1_grad/mulKOptimizer_1/gradients/output_layer_1/truediv_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
=Optimizer_1/gradients/output_layer_1/truediv_1_grad/Reshape_1Reshape9Optimizer_1/gradients/output_layer_1/truediv_1_grad/Sum_1;Optimizer_1/gradients/output_layer_1/truediv_1_grad/Shape_1*
T0*
Tshape0
�
DOptimizer_1/gradients/output_layer_1/truediv_1_grad/tuple/group_depsNoOp<^Optimizer_1/gradients/output_layer_1/truediv_1_grad/Reshape>^Optimizer_1/gradients/output_layer_1/truediv_1_grad/Reshape_1
�
LOptimizer_1/gradients/output_layer_1/truediv_1_grad/tuple/control_dependencyIdentity;Optimizer_1/gradients/output_layer_1/truediv_1_grad/ReshapeE^Optimizer_1/gradients/output_layer_1/truediv_1_grad/tuple/group_deps*
T0*N
_classD
B@loc:@Optimizer_1/gradients/output_layer_1/truediv_1_grad/Reshape
�
NOptimizer_1/gradients/output_layer_1/truediv_1_grad/tuple/control_dependency_1Identity=Optimizer_1/gradients/output_layer_1/truediv_1_grad/Reshape_1E^Optimizer_1/gradients/output_layer_1/truediv_1_grad/tuple/group_deps*
T0*P
_classF
DBloc:@Optimizer_1/gradients/output_layer_1/truediv_1_grad/Reshape_1
b
9Optimizer_1/gradients/output_layer_1/truediv_2_grad/ShapeConst*
dtype0*
valueB 
d
;Optimizer_1/gradients/output_layer_1/truediv_2_grad/Shape_1Const*
dtype0*
valueB 
�
IOptimizer_1/gradients/output_layer_1/truediv_2_grad/BroadcastGradientArgsBroadcastGradientArgs9Optimizer_1/gradients/output_layer_1/truediv_2_grad/Shape;Optimizer_1/gradients/output_layer_1/truediv_2_grad/Shape_1*
T0
�
;Optimizer_1/gradients/output_layer_1/truediv_2_grad/RealDivRealDivLOptimizer_1/gradients/output_layer_1/truediv_3_grad/tuple/control_dependencyoutput_layer_1/Sum_1*
T0
�
7Optimizer_1/gradients/output_layer_1/truediv_2_grad/SumSum;Optimizer_1/gradients/output_layer_1/truediv_2_grad/RealDivIOptimizer_1/gradients/output_layer_1/truediv_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
;Optimizer_1/gradients/output_layer_1/truediv_2_grad/ReshapeReshape7Optimizer_1/gradients/output_layer_1/truediv_2_grad/Sum9Optimizer_1/gradients/output_layer_1/truediv_2_grad/Shape*
T0*
Tshape0
]
7Optimizer_1/gradients/output_layer_1/truediv_2_grad/NegNegoutput_layer_1/Sum_3*
T0
�
=Optimizer_1/gradients/output_layer_1/truediv_2_grad/RealDiv_1RealDiv7Optimizer_1/gradients/output_layer_1/truediv_2_grad/Negoutput_layer_1/Sum_1*
T0
�
=Optimizer_1/gradients/output_layer_1/truediv_2_grad/RealDiv_2RealDiv=Optimizer_1/gradients/output_layer_1/truediv_2_grad/RealDiv_1output_layer_1/Sum_1*
T0
�
7Optimizer_1/gradients/output_layer_1/truediv_2_grad/mulMulLOptimizer_1/gradients/output_layer_1/truediv_3_grad/tuple/control_dependency=Optimizer_1/gradients/output_layer_1/truediv_2_grad/RealDiv_2*
T0
�
9Optimizer_1/gradients/output_layer_1/truediv_2_grad/Sum_1Sum7Optimizer_1/gradients/output_layer_1/truediv_2_grad/mulKOptimizer_1/gradients/output_layer_1/truediv_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
=Optimizer_1/gradients/output_layer_1/truediv_2_grad/Reshape_1Reshape9Optimizer_1/gradients/output_layer_1/truediv_2_grad/Sum_1;Optimizer_1/gradients/output_layer_1/truediv_2_grad/Shape_1*
T0*
Tshape0
�
DOptimizer_1/gradients/output_layer_1/truediv_2_grad/tuple/group_depsNoOp<^Optimizer_1/gradients/output_layer_1/truediv_2_grad/Reshape>^Optimizer_1/gradients/output_layer_1/truediv_2_grad/Reshape_1
�
LOptimizer_1/gradients/output_layer_1/truediv_2_grad/tuple/control_dependencyIdentity;Optimizer_1/gradients/output_layer_1/truediv_2_grad/ReshapeE^Optimizer_1/gradients/output_layer_1/truediv_2_grad/tuple/group_deps*
T0*N
_classD
B@loc:@Optimizer_1/gradients/output_layer_1/truediv_2_grad/Reshape
�
NOptimizer_1/gradients/output_layer_1/truediv_2_grad/tuple/control_dependency_1Identity=Optimizer_1/gradients/output_layer_1/truediv_2_grad/Reshape_1E^Optimizer_1/gradients/output_layer_1/truediv_2_grad/tuple/group_deps*
T0*P
_classF
DBloc:@Optimizer_1/gradients/output_layer_1/truediv_2_grad/Reshape_1
�
=Optimizer_1/gradients/output_layer_1/logistic_loss_grad/ShapeShape output_layer_1/logistic_loss/sub*
T0*
out_type0
�
?Optimizer_1/gradients/output_layer_1/logistic_loss_grad/Shape_1Shape"output_layer_1/logistic_loss/Log1p*
T0*
out_type0
�
MOptimizer_1/gradients/output_layer_1/logistic_loss_grad/BroadcastGradientArgsBroadcastGradientArgs=Optimizer_1/gradients/output_layer_1/logistic_loss_grad/Shape?Optimizer_1/gradients/output_layer_1/logistic_loss_grad/Shape_1*
T0
�
;Optimizer_1/gradients/output_layer_1/logistic_loss_grad/SumSum6Optimizer_1/gradients/output_layer_1/Mean_grad/truedivMOptimizer_1/gradients/output_layer_1/logistic_loss_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
?Optimizer_1/gradients/output_layer_1/logistic_loss_grad/ReshapeReshape;Optimizer_1/gradients/output_layer_1/logistic_loss_grad/Sum=Optimizer_1/gradients/output_layer_1/logistic_loss_grad/Shape*
T0*
Tshape0
�
=Optimizer_1/gradients/output_layer_1/logistic_loss_grad/Sum_1Sum6Optimizer_1/gradients/output_layer_1/Mean_grad/truedivOOptimizer_1/gradients/output_layer_1/logistic_loss_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
AOptimizer_1/gradients/output_layer_1/logistic_loss_grad/Reshape_1Reshape=Optimizer_1/gradients/output_layer_1/logistic_loss_grad/Sum_1?Optimizer_1/gradients/output_layer_1/logistic_loss_grad/Shape_1*
T0*
Tshape0
�
HOptimizer_1/gradients/output_layer_1/logistic_loss_grad/tuple/group_depsNoOp@^Optimizer_1/gradients/output_layer_1/logistic_loss_grad/ReshapeB^Optimizer_1/gradients/output_layer_1/logistic_loss_grad/Reshape_1
�
POptimizer_1/gradients/output_layer_1/logistic_loss_grad/tuple/control_dependencyIdentity?Optimizer_1/gradients/output_layer_1/logistic_loss_grad/ReshapeI^Optimizer_1/gradients/output_layer_1/logistic_loss_grad/tuple/group_deps*
T0*R
_classH
FDloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_grad/Reshape
�
ROptimizer_1/gradients/output_layer_1/logistic_loss_grad/tuple/control_dependency_1IdentityAOptimizer_1/gradients/output_layer_1/logistic_loss_grad/Reshape_1I^Optimizer_1/gradients/output_layer_1/logistic_loss_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_grad/Reshape_1
l
>Optimizer_1/gradients/output_layer_1/Mean_1_grad/Reshape/shapeConst*
dtype0*
valueB:
�
8Optimizer_1/gradients/output_layer_1/Mean_1_grad/ReshapeReshapeHOptimizer_1/gradients/output_layer_1/mul_grad/tuple/control_dependency_1>Optimizer_1/gradients/output_layer_1/Mean_1_grad/Reshape/shape*
T0*
Tshape0
x
6Optimizer_1/gradients/output_layer_1/Mean_1_grad/ShapeShapeoutput_layer_1/logistic_loss_1*
T0*
out_type0
�
5Optimizer_1/gradients/output_layer_1/Mean_1_grad/TileTile8Optimizer_1/gradients/output_layer_1/Mean_1_grad/Reshape6Optimizer_1/gradients/output_layer_1/Mean_1_grad/Shape*
T0*

Tmultiples0
z
8Optimizer_1/gradients/output_layer_1/Mean_1_grad/Shape_1Shapeoutput_layer_1/logistic_loss_1*
T0*
out_type0
a
8Optimizer_1/gradients/output_layer_1/Mean_1_grad/Shape_2Const*
dtype0*
valueB 
d
6Optimizer_1/gradients/output_layer_1/Mean_1_grad/ConstConst*
dtype0*
valueB: 
�
5Optimizer_1/gradients/output_layer_1/Mean_1_grad/ProdProd8Optimizer_1/gradients/output_layer_1/Mean_1_grad/Shape_16Optimizer_1/gradients/output_layer_1/Mean_1_grad/Const*
T0*

Tidx0*
	keep_dims( 
f
8Optimizer_1/gradients/output_layer_1/Mean_1_grad/Const_1Const*
dtype0*
valueB: 
�
7Optimizer_1/gradients/output_layer_1/Mean_1_grad/Prod_1Prod8Optimizer_1/gradients/output_layer_1/Mean_1_grad/Shape_28Optimizer_1/gradients/output_layer_1/Mean_1_grad/Const_1*
T0*

Tidx0*
	keep_dims( 
d
:Optimizer_1/gradients/output_layer_1/Mean_1_grad/Maximum/yConst*
dtype0*
value	B :
�
8Optimizer_1/gradients/output_layer_1/Mean_1_grad/MaximumMaximum7Optimizer_1/gradients/output_layer_1/Mean_1_grad/Prod_1:Optimizer_1/gradients/output_layer_1/Mean_1_grad/Maximum/y*
T0
�
9Optimizer_1/gradients/output_layer_1/Mean_1_grad/floordivFloorDiv5Optimizer_1/gradients/output_layer_1/Mean_1_grad/Prod8Optimizer_1/gradients/output_layer_1/Mean_1_grad/Maximum*
T0
�
5Optimizer_1/gradients/output_layer_1/Mean_1_grad/CastCast9Optimizer_1/gradients/output_layer_1/Mean_1_grad/floordiv*

DstT0*

SrcT0*
Truncate( 
�
8Optimizer_1/gradients/output_layer_1/Mean_1_grad/truedivRealDiv5Optimizer_1/gradients/output_layer_1/Mean_1_grad/Tile5Optimizer_1/gradients/output_layer_1/Mean_1_grad/Cast*
T0
`
7Optimizer_1/gradients/output_layer_1/truediv_grad/ShapeConst*
dtype0*
valueB 
b
9Optimizer_1/gradients/output_layer_1/truediv_grad/Shape_1Const*
dtype0*
valueB 
�
GOptimizer_1/gradients/output_layer_1/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs7Optimizer_1/gradients/output_layer_1/truediv_grad/Shape9Optimizer_1/gradients/output_layer_1/truediv_grad/Shape_1*
T0
�
9Optimizer_1/gradients/output_layer_1/truediv_grad/RealDivRealDivLOptimizer_1/gradients/output_layer_1/truediv_1_grad/tuple/control_dependencyoutput_layer_1/Sum_1*
T0
�
5Optimizer_1/gradients/output_layer_1/truediv_grad/SumSum9Optimizer_1/gradients/output_layer_1/truediv_grad/RealDivGOptimizer_1/gradients/output_layer_1/truediv_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
9Optimizer_1/gradients/output_layer_1/truediv_grad/ReshapeReshape5Optimizer_1/gradients/output_layer_1/truediv_grad/Sum7Optimizer_1/gradients/output_layer_1/truediv_grad/Shape*
T0*
Tshape0
[
5Optimizer_1/gradients/output_layer_1/truediv_grad/NegNegoutput_layer_1/Sum_2*
T0
�
;Optimizer_1/gradients/output_layer_1/truediv_grad/RealDiv_1RealDiv5Optimizer_1/gradients/output_layer_1/truediv_grad/Negoutput_layer_1/Sum_1*
T0
�
;Optimizer_1/gradients/output_layer_1/truediv_grad/RealDiv_2RealDiv;Optimizer_1/gradients/output_layer_1/truediv_grad/RealDiv_1output_layer_1/Sum_1*
T0
�
5Optimizer_1/gradients/output_layer_1/truediv_grad/mulMulLOptimizer_1/gradients/output_layer_1/truediv_1_grad/tuple/control_dependency;Optimizer_1/gradients/output_layer_1/truediv_grad/RealDiv_2*
T0
�
7Optimizer_1/gradients/output_layer_1/truediv_grad/Sum_1Sum5Optimizer_1/gradients/output_layer_1/truediv_grad/mulIOptimizer_1/gradients/output_layer_1/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
;Optimizer_1/gradients/output_layer_1/truediv_grad/Reshape_1Reshape7Optimizer_1/gradients/output_layer_1/truediv_grad/Sum_19Optimizer_1/gradients/output_layer_1/truediv_grad/Shape_1*
T0*
Tshape0
�
BOptimizer_1/gradients/output_layer_1/truediv_grad/tuple/group_depsNoOp:^Optimizer_1/gradients/output_layer_1/truediv_grad/Reshape<^Optimizer_1/gradients/output_layer_1/truediv_grad/Reshape_1
�
JOptimizer_1/gradients/output_layer_1/truediv_grad/tuple/control_dependencyIdentity9Optimizer_1/gradients/output_layer_1/truediv_grad/ReshapeC^Optimizer_1/gradients/output_layer_1/truediv_grad/tuple/group_deps*
T0*L
_classB
@>loc:@Optimizer_1/gradients/output_layer_1/truediv_grad/Reshape
�
LOptimizer_1/gradients/output_layer_1/truediv_grad/tuple/control_dependency_1Identity;Optimizer_1/gradients/output_layer_1/truediv_grad/Reshape_1C^Optimizer_1/gradients/output_layer_1/truediv_grad/tuple/group_deps*
T0*N
_classD
B@loc:@Optimizer_1/gradients/output_layer_1/truediv_grad/Reshape_1
v
=Optimizer_1/gradients/output_layer_1/Sum_3_grad/Reshape/shapeConst*
dtype0*!
valueB"         
�
7Optimizer_1/gradients/output_layer_1/Sum_3_grad/ReshapeReshapeLOptimizer_1/gradients/output_layer_1/truediv_2_grad/tuple/control_dependency=Optimizer_1/gradients/output_layer_1/Sum_3_grad/Reshape/shape*
T0*
Tshape0
n
5Optimizer_1/gradients/output_layer_1/Sum_3_grad/ShapeShapeoutput_layer_1/Square*
T0*
out_type0
�
4Optimizer_1/gradients/output_layer_1/Sum_3_grad/TileTile7Optimizer_1/gradients/output_layer_1/Sum_3_grad/Reshape5Optimizer_1/gradients/output_layer_1/Sum_3_grad/Shape*
T0*

Tmultiples0
�
AOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/ShapeShape#output_layer_1/logistic_loss/Select*
T0*
out_type0
�
COptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/Shape_1Shape output_layer_1/logistic_loss/mul*
T0*
out_type0
�
QOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/BroadcastGradientArgsBroadcastGradientArgsAOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/ShapeCOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/Shape_1*
T0
�
?Optimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/SumSumPOptimizer_1/gradients/output_layer_1/logistic_loss_grad/tuple/control_dependencyQOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
COptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/ReshapeReshape?Optimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/SumAOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/Shape*
T0*
Tshape0
�
?Optimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/NegNegPOptimizer_1/gradients/output_layer_1/logistic_loss_grad/tuple/control_dependency*
T0
�
AOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/Sum_1Sum?Optimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/NegSOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
EOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/Reshape_1ReshapeAOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/Sum_1COptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/Shape_1*
T0*
Tshape0
�
LOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/tuple/group_depsNoOpD^Optimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/ReshapeF^Optimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/Reshape_1
�
TOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/tuple/control_dependencyIdentityCOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/ReshapeM^Optimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/Reshape
�
VOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/tuple/control_dependency_1IdentityEOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/Reshape_1M^Optimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/Reshape_1
�
COptimizer_1/gradients/output_layer_1/logistic_loss/Log1p_grad/add/xConstS^Optimizer_1/gradients/output_layer_1/logistic_loss_grad/tuple/control_dependency_1*
dtype0*
valueB
 *  �?
�
AOptimizer_1/gradients/output_layer_1/logistic_loss/Log1p_grad/addAddV2COptimizer_1/gradients/output_layer_1/logistic_loss/Log1p_grad/add/x output_layer_1/logistic_loss/Exp*
T0
�
HOptimizer_1/gradients/output_layer_1/logistic_loss/Log1p_grad/Reciprocal
ReciprocalAOptimizer_1/gradients/output_layer_1/logistic_loss/Log1p_grad/add*
T0
�
AOptimizer_1/gradients/output_layer_1/logistic_loss/Log1p_grad/mulMulROptimizer_1/gradients/output_layer_1/logistic_loss_grad/tuple/control_dependency_1HOptimizer_1/gradients/output_layer_1/logistic_loss/Log1p_grad/Reciprocal*
T0
�
?Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/ShapeShape"output_layer_1/logistic_loss_1/sub*
T0*
out_type0
�
AOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Shape_1Shape$output_layer_1/logistic_loss_1/Log1p*
T0*
out_type0
�
OOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/BroadcastGradientArgsBroadcastGradientArgs?Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/ShapeAOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Shape_1*
T0
�
=Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/SumSum8Optimizer_1/gradients/output_layer_1/Mean_1_grad/truedivOOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
AOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/ReshapeReshape=Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Sum?Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Shape*
T0*
Tshape0
�
?Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Sum_1Sum8Optimizer_1/gradients/output_layer_1/Mean_1_grad/truedivQOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
COptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Reshape_1Reshape?Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Sum_1AOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Shape_1*
T0*
Tshape0
�
JOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/tuple/group_depsNoOpB^Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/ReshapeD^Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Reshape_1
�
ROptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/tuple/control_dependencyIdentityAOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/ReshapeK^Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/tuple/group_deps*
T0*T
_classJ
HFloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Reshape
�
TOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/tuple/control_dependency_1IdentityCOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Reshape_1K^Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/Reshape_1
v
=Optimizer_1/gradients/output_layer_1/Sum_2_grad/Reshape/shapeConst*
dtype0*!
valueB"         
�
7Optimizer_1/gradients/output_layer_1/Sum_2_grad/ReshapeReshapeJOptimizer_1/gradients/output_layer_1/truediv_grad/tuple/control_dependency=Optimizer_1/gradients/output_layer_1/Sum_2_grad/Reshape/shape*
T0*
Tshape0
m
5Optimizer_1/gradients/output_layer_1/Sum_2_grad/ShapeShapeoutput_layer_1/Abs_1*
T0*
out_type0
�
4Optimizer_1/gradients/output_layer_1/Sum_2_grad/TileTile7Optimizer_1/gradients/output_layer_1/Sum_2_grad/Reshape5Optimizer_1/gradients/output_layer_1/Sum_2_grad/Shape*
T0*

Tmultiples0
�
6Optimizer_1/gradients/output_layer_1/Square_grad/ConstConst5^Optimizer_1/gradients/output_layer_1/Sum_3_grad/Tile*
dtype0*
valueB
 *   @
�
4Optimizer_1/gradients/output_layer_1/Square_grad/MulMuloutput_layer_1/sub_16Optimizer_1/gradients/output_layer_1/Square_grad/Const*
T0
�
6Optimizer_1/gradients/output_layer_1/Square_grad/Mul_1Mul4Optimizer_1/gradients/output_layer_1/Sum_3_grad/Tile4Optimizer_1/gradients/output_layer_1/Square_grad/Mul*
T0
x
IOptimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/zeros_like	ZerosLikeoutput_layer_1/GatherNd*
T0
�
EOptimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/SelectSelect)output_layer_1/logistic_loss/GreaterEqualTOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/tuple/control_dependencyIOptimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/zeros_like*
T0
�
GOptimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/Select_1Select)output_layer_1/logistic_loss/GreaterEqualIOptimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/zeros_likeTOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/tuple/control_dependency*
T0
�
OOptimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/tuple/group_depsNoOpF^Optimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/SelectH^Optimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/Select_1
�
WOptimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/tuple/control_dependencyIdentityEOptimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/SelectP^Optimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/Select
�
YOptimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/tuple/control_dependency_1IdentityGOptimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/Select_1P^Optimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/Select_1
|
AOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/ShapeShapeoutput_layer_1/GatherNd*
T0*
out_type0
�
COptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Shape_1Shapeoutput_layer_1/GatherNd_2*
T0*
out_type0
�
QOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/BroadcastGradientArgsBroadcastGradientArgsAOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/ShapeCOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Shape_1*
T0
�
?Optimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/MulMulVOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/tuple/control_dependency_1output_layer_1/GatherNd_2*
T0
�
?Optimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/SumSum?Optimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/MulQOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
COptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/ReshapeReshape?Optimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/SumAOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Shape*
T0*
Tshape0
�
AOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Mul_1Muloutput_layer_1/GatherNdVOptimizer_1/gradients/output_layer_1/logistic_loss/sub_grad/tuple/control_dependency_1*
T0
�
AOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Sum_1SumAOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Mul_1SOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
EOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Reshape_1ReshapeAOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Sum_1COptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Shape_1*
T0*
Tshape0
�
LOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/tuple/group_depsNoOpD^Optimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/ReshapeF^Optimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Reshape_1
�
TOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/tuple/control_dependencyIdentityCOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/ReshapeM^Optimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Reshape
�
VOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/tuple/control_dependency_1IdentityEOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Reshape_1M^Optimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/Reshape_1
�
?Optimizer_1/gradients/output_layer_1/logistic_loss/Exp_grad/mulMulAOptimizer_1/gradients/output_layer_1/logistic_loss/Log1p_grad/mul output_layer_1/logistic_loss/Exp*
T0
�
COptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/ShapeShape%output_layer_1/logistic_loss_1/Select*
T0*
out_type0
�
EOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/Shape_1Shape"output_layer_1/logistic_loss_1/mul*
T0*
out_type0
�
SOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgsCOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/ShapeEOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/Shape_1*
T0
�
AOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/SumSumROptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/tuple/control_dependencySOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
EOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/ReshapeReshapeAOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/SumCOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/Shape*
T0*
Tshape0
�
AOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/NegNegROptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/tuple/control_dependency*
T0
�
COptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/Sum_1SumAOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/NegUOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
GOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/Reshape_1ReshapeCOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/Sum_1EOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/Shape_1*
T0*
Tshape0
�
NOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/tuple/group_depsNoOpF^Optimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/ReshapeH^Optimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/Reshape_1
�
VOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/tuple/control_dependencyIdentityEOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/ReshapeO^Optimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/Reshape
�
XOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/tuple/control_dependency_1IdentityGOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/Reshape_1O^Optimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/Reshape_1
�
EOptimizer_1/gradients/output_layer_1/logistic_loss_1/Log1p_grad/add/xConstU^Optimizer_1/gradients/output_layer_1/logistic_loss_1_grad/tuple/control_dependency_1*
dtype0*
valueB
 *  �?
�
COptimizer_1/gradients/output_layer_1/logistic_loss_1/Log1p_grad/addAddV2EOptimizer_1/gradients/output_layer_1/logistic_loss_1/Log1p_grad/add/x"output_layer_1/logistic_loss_1/Exp*
T0
�
JOptimizer_1/gradients/output_layer_1/logistic_loss_1/Log1p_grad/Reciprocal
ReciprocalCOptimizer_1/gradients/output_layer_1/logistic_loss_1/Log1p_grad/add*
T0
�
COptimizer_1/gradients/output_layer_1/logistic_loss_1/Log1p_grad/mulMulTOptimizer_1/gradients/output_layer_1/logistic_loss_1_grad/tuple/control_dependency_1JOptimizer_1/gradients/output_layer_1/logistic_loss_1/Log1p_grad/Reciprocal*
T0
Y
4Optimizer_1/gradients/output_layer_1/Abs_1_grad/SignSignoutput_layer_1/sub*
T0
�
3Optimizer_1/gradients/output_layer_1/Abs_1_grad/mulMul4Optimizer_1/gradients/output_layer_1/Sum_2_grad/Tile4Optimizer_1/gradients/output_layer_1/Abs_1_grad/Sign*
T0
w
5Optimizer_1/gradients/output_layer_1/sub_1_grad/ShapeShapeoutput_layer_1/strided_slice_5*
T0*
out_type0
y
7Optimizer_1/gradients/output_layer_1/sub_1_grad/Shape_1Shapeoutput_layer_1/strided_slice_6*
T0*
out_type0
�
EOptimizer_1/gradients/output_layer_1/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgs5Optimizer_1/gradients/output_layer_1/sub_1_grad/Shape7Optimizer_1/gradients/output_layer_1/sub_1_grad/Shape_1*
T0
�
3Optimizer_1/gradients/output_layer_1/sub_1_grad/SumSum6Optimizer_1/gradients/output_layer_1/Square_grad/Mul_1EOptimizer_1/gradients/output_layer_1/sub_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
7Optimizer_1/gradients/output_layer_1/sub_1_grad/ReshapeReshape3Optimizer_1/gradients/output_layer_1/sub_1_grad/Sum5Optimizer_1/gradients/output_layer_1/sub_1_grad/Shape*
T0*
Tshape0
{
3Optimizer_1/gradients/output_layer_1/sub_1_grad/NegNeg6Optimizer_1/gradients/output_layer_1/Square_grad/Mul_1*
T0
�
5Optimizer_1/gradients/output_layer_1/sub_1_grad/Sum_1Sum3Optimizer_1/gradients/output_layer_1/sub_1_grad/NegGOptimizer_1/gradients/output_layer_1/sub_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
9Optimizer_1/gradients/output_layer_1/sub_1_grad/Reshape_1Reshape5Optimizer_1/gradients/output_layer_1/sub_1_grad/Sum_17Optimizer_1/gradients/output_layer_1/sub_1_grad/Shape_1*
T0*
Tshape0
�
@Optimizer_1/gradients/output_layer_1/sub_1_grad/tuple/group_depsNoOp8^Optimizer_1/gradients/output_layer_1/sub_1_grad/Reshape:^Optimizer_1/gradients/output_layer_1/sub_1_grad/Reshape_1
�
HOptimizer_1/gradients/output_layer_1/sub_1_grad/tuple/control_dependencyIdentity7Optimizer_1/gradients/output_layer_1/sub_1_grad/ReshapeA^Optimizer_1/gradients/output_layer_1/sub_1_grad/tuple/group_deps*
T0*J
_class@
><loc:@Optimizer_1/gradients/output_layer_1/sub_1_grad/Reshape
�
JOptimizer_1/gradients/output_layer_1/sub_1_grad/tuple/control_dependency_1Identity9Optimizer_1/gradients/output_layer_1/sub_1_grad/Reshape_1A^Optimizer_1/gradients/output_layer_1/sub_1_grad/tuple/group_deps*
T0*L
_classB
@>loc:@Optimizer_1/gradients/output_layer_1/sub_1_grad/Reshape_1
�
KOptimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/zeros_like	ZerosLike output_layer_1/logistic_loss/Neg*
T0
�
GOptimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/SelectSelect)output_layer_1/logistic_loss/GreaterEqual?Optimizer_1/gradients/output_layer_1/logistic_loss/Exp_grad/mulKOptimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/zeros_like*
T0
�
IOptimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/Select_1Select)output_layer_1/logistic_loss/GreaterEqualKOptimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/zeros_like?Optimizer_1/gradients/output_layer_1/logistic_loss/Exp_grad/mul*
T0
�
QOptimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/tuple/group_depsNoOpH^Optimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/SelectJ^Optimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/Select_1
�
YOptimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/tuple/control_dependencyIdentityGOptimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/SelectR^Optimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/Select
�
[Optimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/tuple/control_dependency_1IdentityIOptimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/Select_1R^Optimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/Select_1
|
KOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/zeros_like	ZerosLikeoutput_layer_1/GatherNd_3*
T0
�
GOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/SelectSelect+output_layer_1/logistic_loss_1/GreaterEqualVOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/tuple/control_dependencyKOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/zeros_like*
T0
�
IOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/Select_1Select+output_layer_1/logistic_loss_1/GreaterEqualKOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/zeros_likeVOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/tuple/control_dependency*
T0
�
QOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/tuple/group_depsNoOpH^Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/SelectJ^Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/Select_1
�
YOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/tuple/control_dependencyIdentityGOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/SelectR^Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/Select
�
[Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/tuple/control_dependency_1IdentityIOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/Select_1R^Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/Select_1
�
COptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/ShapeShapeoutput_layer_1/GatherNd_3*
T0*
out_type0
�
EOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Shape_1Shapeoutput_layer_1/GatherNd_5*
T0*
out_type0
�
SOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgsCOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/ShapeEOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Shape_1*
T0
�
AOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/MulMulXOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/tuple/control_dependency_1output_layer_1/GatherNd_5*
T0
�
AOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/SumSumAOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/MulSOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
EOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/ReshapeReshapeAOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/SumCOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Shape*
T0*
Tshape0
�
COptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Mul_1Muloutput_layer_1/GatherNd_3XOptimizer_1/gradients/output_layer_1/logistic_loss_1/sub_grad/tuple/control_dependency_1*
T0
�
COptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Sum_1SumCOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Mul_1UOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
GOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Reshape_1ReshapeCOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Sum_1EOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Shape_1*
T0*
Tshape0
�
NOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/tuple/group_depsNoOpF^Optimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/ReshapeH^Optimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Reshape_1
�
VOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/tuple/control_dependencyIdentityEOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/ReshapeO^Optimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Reshape
�
XOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/tuple/control_dependency_1IdentityGOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Reshape_1O^Optimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/Reshape_1
�
AOptimizer_1/gradients/output_layer_1/logistic_loss_1/Exp_grad/mulMulCOptimizer_1/gradients/output_layer_1/logistic_loss_1/Log1p_grad/mul"output_layer_1/logistic_loss_1/Exp*
T0
u
3Optimizer_1/gradients/output_layer_1/sub_grad/ShapeShapeoutput_layer_1/strided_slice_3*
T0*
out_type0
w
5Optimizer_1/gradients/output_layer_1/sub_grad/Shape_1Shapeoutput_layer_1/strided_slice_4*
T0*
out_type0
�
COptimizer_1/gradients/output_layer_1/sub_grad/BroadcastGradientArgsBroadcastGradientArgs3Optimizer_1/gradients/output_layer_1/sub_grad/Shape5Optimizer_1/gradients/output_layer_1/sub_grad/Shape_1*
T0
�
1Optimizer_1/gradients/output_layer_1/sub_grad/SumSum3Optimizer_1/gradients/output_layer_1/Abs_1_grad/mulCOptimizer_1/gradients/output_layer_1/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
5Optimizer_1/gradients/output_layer_1/sub_grad/ReshapeReshape1Optimizer_1/gradients/output_layer_1/sub_grad/Sum3Optimizer_1/gradients/output_layer_1/sub_grad/Shape*
T0*
Tshape0
v
1Optimizer_1/gradients/output_layer_1/sub_grad/NegNeg3Optimizer_1/gradients/output_layer_1/Abs_1_grad/mul*
T0
�
3Optimizer_1/gradients/output_layer_1/sub_grad/Sum_1Sum1Optimizer_1/gradients/output_layer_1/sub_grad/NegEOptimizer_1/gradients/output_layer_1/sub_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
7Optimizer_1/gradients/output_layer_1/sub_grad/Reshape_1Reshape3Optimizer_1/gradients/output_layer_1/sub_grad/Sum_15Optimizer_1/gradients/output_layer_1/sub_grad/Shape_1*
T0*
Tshape0
�
>Optimizer_1/gradients/output_layer_1/sub_grad/tuple/group_depsNoOp6^Optimizer_1/gradients/output_layer_1/sub_grad/Reshape8^Optimizer_1/gradients/output_layer_1/sub_grad/Reshape_1
�
FOptimizer_1/gradients/output_layer_1/sub_grad/tuple/control_dependencyIdentity5Optimizer_1/gradients/output_layer_1/sub_grad/Reshape?^Optimizer_1/gradients/output_layer_1/sub_grad/tuple/group_deps*
T0*H
_class>
<:loc:@Optimizer_1/gradients/output_layer_1/sub_grad/Reshape
�
HOptimizer_1/gradients/output_layer_1/sub_grad/tuple/control_dependency_1Identity7Optimizer_1/gradients/output_layer_1/sub_grad/Reshape_1?^Optimizer_1/gradients/output_layer_1/sub_grad/tuple/group_deps*
T0*J
_class@
><loc:@Optimizer_1/gradients/output_layer_1/sub_grad/Reshape_1
w
?Optimizer_1/gradients/output_layer_1/strided_slice_5_grad/ShapeShapeoutput_layer_1/preds*
T0*
out_type0
�
POptimizer_1/gradients/output_layer_1/strided_slice_5_grad/StridedSliceGrad/beginConst*
dtype0*!
valueB"           
�
NOptimizer_1/gradients/output_layer_1/strided_slice_5_grad/StridedSliceGrad/endConst*
dtype0*!
valueB"            
�
ROptimizer_1/gradients/output_layer_1/strided_slice_5_grad/StridedSliceGrad/stridesConst*
dtype0*!
valueB"         
�
JOptimizer_1/gradients/output_layer_1/strided_slice_5_grad/StridedSliceGradStridedSliceGrad?Optimizer_1/gradients/output_layer_1/strided_slice_5_grad/ShapePOptimizer_1/gradients/output_layer_1/strided_slice_5_grad/StridedSliceGrad/beginNOptimizer_1/gradients/output_layer_1/strided_slice_5_grad/StridedSliceGrad/endROptimizer_1/gradients/output_layer_1/strided_slice_5_grad/StridedSliceGrad/stridesHOptimizer_1/gradients/output_layer_1/sub_1_grad/tuple/control_dependency*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
w
?Optimizer_1/gradients/output_layer_1/strided_slice_6_grad/ShapeShapeoutput_layer_1/preds*
T0*
out_type0
�
POptimizer_1/gradients/output_layer_1/strided_slice_6_grad/StridedSliceGrad/beginConst*
dtype0*!
valueB"            
�
NOptimizer_1/gradients/output_layer_1/strided_slice_6_grad/StridedSliceGrad/endConst*
dtype0*!
valueB"    ����    
�
ROptimizer_1/gradients/output_layer_1/strided_slice_6_grad/StridedSliceGrad/stridesConst*
dtype0*!
valueB"         
�
JOptimizer_1/gradients/output_layer_1/strided_slice_6_grad/StridedSliceGradStridedSliceGrad?Optimizer_1/gradients/output_layer_1/strided_slice_6_grad/ShapePOptimizer_1/gradients/output_layer_1/strided_slice_6_grad/StridedSliceGrad/beginNOptimizer_1/gradients/output_layer_1/strided_slice_6_grad/StridedSliceGrad/endROptimizer_1/gradients/output_layer_1/strided_slice_6_grad/StridedSliceGrad/stridesJOptimizer_1/gradients/output_layer_1/sub_1_grad/tuple/control_dependency_1*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
�
?Optimizer_1/gradients/output_layer_1/logistic_loss/Neg_grad/NegNegYOptimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/tuple/control_dependency*
T0
�
MOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/zeros_like	ZerosLike"output_layer_1/logistic_loss_1/Neg*
T0
�
IOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/SelectSelect+output_layer_1/logistic_loss_1/GreaterEqualAOptimizer_1/gradients/output_layer_1/logistic_loss_1/Exp_grad/mulMOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/zeros_like*
T0
�
KOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/Select_1Select+output_layer_1/logistic_loss_1/GreaterEqualMOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/zeros_likeAOptimizer_1/gradients/output_layer_1/logistic_loss_1/Exp_grad/mul*
T0
�
SOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/tuple/group_depsNoOpJ^Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/SelectL^Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/Select_1
�
[Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/tuple/control_dependencyIdentityIOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/SelectT^Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/Select
�
]Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/tuple/control_dependency_1IdentityKOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/Select_1T^Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/tuple/group_deps*
T0*^
_classT
RPloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/Select_1
w
?Optimizer_1/gradients/output_layer_1/strided_slice_3_grad/ShapeShapeoutput_layer_1/preds*
T0*
out_type0
�
POptimizer_1/gradients/output_layer_1/strided_slice_3_grad/StridedSliceGrad/beginConst*
dtype0*!
valueB"           
�
NOptimizer_1/gradients/output_layer_1/strided_slice_3_grad/StridedSliceGrad/endConst*
dtype0*!
valueB"            
�
ROptimizer_1/gradients/output_layer_1/strided_slice_3_grad/StridedSliceGrad/stridesConst*
dtype0*!
valueB"         
�
JOptimizer_1/gradients/output_layer_1/strided_slice_3_grad/StridedSliceGradStridedSliceGrad?Optimizer_1/gradients/output_layer_1/strided_slice_3_grad/ShapePOptimizer_1/gradients/output_layer_1/strided_slice_3_grad/StridedSliceGrad/beginNOptimizer_1/gradients/output_layer_1/strided_slice_3_grad/StridedSliceGrad/endROptimizer_1/gradients/output_layer_1/strided_slice_3_grad/StridedSliceGrad/stridesFOptimizer_1/gradients/output_layer_1/sub_grad/tuple/control_dependency*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
w
?Optimizer_1/gradients/output_layer_1/strided_slice_4_grad/ShapeShapeoutput_layer_1/preds*
T0*
out_type0
�
POptimizer_1/gradients/output_layer_1/strided_slice_4_grad/StridedSliceGrad/beginConst*
dtype0*!
valueB"            
�
NOptimizer_1/gradients/output_layer_1/strided_slice_4_grad/StridedSliceGrad/endConst*
dtype0*!
valueB"    ����    
�
ROptimizer_1/gradients/output_layer_1/strided_slice_4_grad/StridedSliceGrad/stridesConst*
dtype0*!
valueB"         
�
JOptimizer_1/gradients/output_layer_1/strided_slice_4_grad/StridedSliceGradStridedSliceGrad?Optimizer_1/gradients/output_layer_1/strided_slice_4_grad/ShapePOptimizer_1/gradients/output_layer_1/strided_slice_4_grad/StridedSliceGrad/beginNOptimizer_1/gradients/output_layer_1/strided_slice_4_grad/StridedSliceGrad/endROptimizer_1/gradients/output_layer_1/strided_slice_4_grad/StridedSliceGrad/stridesHOptimizer_1/gradients/output_layer_1/sub_grad/tuple/control_dependency_1*
Index0*
T0*

begin_mask*
ellipsis_mask *
end_mask*
new_axis_mask *
shrink_axis_mask 
�
AOptimizer_1/gradients/output_layer_1/logistic_loss_1/Neg_grad/NegNeg[Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/tuple/control_dependency*
T0
�
Optimizer_1/gradients/AddNAddNJOptimizer_1/gradients/output_layer_1/strided_slice_5_grad/StridedSliceGradJOptimizer_1/gradients/output_layer_1/strided_slice_6_grad/StridedSliceGradJOptimizer_1/gradients/output_layer_1/strided_slice_3_grad/StridedSliceGradJOptimizer_1/gradients/output_layer_1/strided_slice_4_grad/StridedSliceGrad*
N*
T0*]
_classS
QOloc:@Optimizer_1/gradients/output_layer_1/strided_slice_5_grad/StridedSliceGrad
�
;Optimizer_1/gradients/output_layer_1/preds_grad/SigmoidGradSigmoidGradoutput_layer_1/predsOptimizer_1/gradients/AddN*
T0
�
Optimizer_1/gradients/AddN_1AddNWOptimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/tuple/control_dependencyTOptimizer_1/gradients/output_layer_1/logistic_loss/mul_grad/tuple/control_dependency[Optimizer_1/gradients/output_layer_1/logistic_loss/Select_1_grad/tuple/control_dependency_1?Optimizer_1/gradients/output_layer_1/logistic_loss/Neg_grad/Neg*
N*
T0*X
_classN
LJloc:@Optimizer_1/gradients/output_layer_1/logistic_loss/Select_grad/Select
t
8Optimizer_1/gradients/output_layer_1/GatherNd_grad/ShapeShapeoutput_layer_1/Reshape_1*
T0*
out_type0	
�
<Optimizer_1/gradients/output_layer_1/GatherNd_grad/ScatterNd	ScatterNdoutput_layer_1/WhereOptimizer_1/gradients/AddN_18Optimizer_1/gradients/output_layer_1/GatherNd_grad/Shape*
T0*
Tindices0	
�
Optimizer_1/gradients/AddN_2AddNYOptimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/tuple/control_dependencyVOptimizer_1/gradients/output_layer_1/logistic_loss_1/mul_grad/tuple/control_dependency]Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_1_grad/tuple/control_dependency_1AOptimizer_1/gradients/output_layer_1/logistic_loss_1/Neg_grad/Neg*
N*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/output_layer_1/logistic_loss_1/Select_grad/Select
v
:Optimizer_1/gradients/output_layer_1/GatherNd_3_grad/ShapeShapeoutput_layer_1/Reshape_1*
T0*
out_type0	
�
>Optimizer_1/gradients/output_layer_1/GatherNd_3_grad/ScatterNd	ScatterNdoutput_layer_1/Where_1Optimizer_1/gradients/AddN_2:Optimizer_1/gradients/output_layer_1/GatherNd_3_grad/Shape*
T0*
Tindices0	
�
Optimizer_1/gradients/AddN_3AddN;Optimizer_1/gradients/output_layer_1/preds_grad/SigmoidGrad<Optimizer_1/gradients/output_layer_1/GatherNd_grad/ScatterNd>Optimizer_1/gradients/output_layer_1/GatherNd_3_grad/ScatterNd*
N*
T0*N
_classD
B@loc:@Optimizer_1/gradients/output_layer_1/preds_grad/SigmoidGrad
o
9Optimizer_1/gradients/output_layer_1/Reshape_1_grad/ShapeShapeoutput_layer_1/add*
T0*
out_type0
�
;Optimizer_1/gradients/output_layer_1/Reshape_1_grad/ReshapeReshapeOptimizer_1/gradients/AddN_39Optimizer_1/gradients/output_layer_1/Reshape_1_grad/Shape*
T0*
Tshape0
l
3Optimizer_1/gradients/output_layer_1/add_grad/ShapeShapeoutput_layer_1/MatMul*
T0*
out_type0
q
5Optimizer_1/gradients/output_layer_1/add_grad/Shape_1Shapeoutput_layer/biases/read*
T0*
out_type0
�
COptimizer_1/gradients/output_layer_1/add_grad/BroadcastGradientArgsBroadcastGradientArgs3Optimizer_1/gradients/output_layer_1/add_grad/Shape5Optimizer_1/gradients/output_layer_1/add_grad/Shape_1*
T0
�
1Optimizer_1/gradients/output_layer_1/add_grad/SumSum;Optimizer_1/gradients/output_layer_1/Reshape_1_grad/ReshapeCOptimizer_1/gradients/output_layer_1/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
5Optimizer_1/gradients/output_layer_1/add_grad/ReshapeReshape1Optimizer_1/gradients/output_layer_1/add_grad/Sum3Optimizer_1/gradients/output_layer_1/add_grad/Shape*
T0*
Tshape0
�
3Optimizer_1/gradients/output_layer_1/add_grad/Sum_1Sum;Optimizer_1/gradients/output_layer_1/Reshape_1_grad/ReshapeEOptimizer_1/gradients/output_layer_1/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
7Optimizer_1/gradients/output_layer_1/add_grad/Reshape_1Reshape3Optimizer_1/gradients/output_layer_1/add_grad/Sum_15Optimizer_1/gradients/output_layer_1/add_grad/Shape_1*
T0*
Tshape0
�
>Optimizer_1/gradients/output_layer_1/add_grad/tuple/group_depsNoOp6^Optimizer_1/gradients/output_layer_1/add_grad/Reshape8^Optimizer_1/gradients/output_layer_1/add_grad/Reshape_1
�
FOptimizer_1/gradients/output_layer_1/add_grad/tuple/control_dependencyIdentity5Optimizer_1/gradients/output_layer_1/add_grad/Reshape?^Optimizer_1/gradients/output_layer_1/add_grad/tuple/group_deps*
T0*H
_class>
<:loc:@Optimizer_1/gradients/output_layer_1/add_grad/Reshape
�
HOptimizer_1/gradients/output_layer_1/add_grad/tuple/control_dependency_1Identity7Optimizer_1/gradients/output_layer_1/add_grad/Reshape_1?^Optimizer_1/gradients/output_layer_1/add_grad/tuple/group_deps*
T0*J
_class@
><loc:@Optimizer_1/gradients/output_layer_1/add_grad/Reshape_1
�
7Optimizer_1/gradients/output_layer_1/MatMul_grad/MatMulMatMulFOptimizer_1/gradients/output_layer_1/add_grad/tuple/control_dependencyoutput_layer/weights/read*
T0*
transpose_a( *
transpose_b(
�
9Optimizer_1/gradients/output_layer_1/MatMul_grad/MatMul_1MatMuloutput_layer_1/ReshapeFOptimizer_1/gradients/output_layer_1/add_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
AOptimizer_1/gradients/output_layer_1/MatMul_grad/tuple/group_depsNoOp8^Optimizer_1/gradients/output_layer_1/MatMul_grad/MatMul:^Optimizer_1/gradients/output_layer_1/MatMul_grad/MatMul_1
�
IOptimizer_1/gradients/output_layer_1/MatMul_grad/tuple/control_dependencyIdentity7Optimizer_1/gradients/output_layer_1/MatMul_grad/MatMulB^Optimizer_1/gradients/output_layer_1/MatMul_grad/tuple/group_deps*
T0*J
_class@
><loc:@Optimizer_1/gradients/output_layer_1/MatMul_grad/MatMul
�
KOptimizer_1/gradients/output_layer_1/MatMul_grad/tuple/control_dependency_1Identity9Optimizer_1/gradients/output_layer_1/MatMul_grad/MatMul_1B^Optimizer_1/gradients/output_layer_1/MatMul_grad/tuple/group_deps*
T0*L
_classB
@>loc:@Optimizer_1/gradients/output_layer_1/MatMul_grad/MatMul_1
{
7Optimizer_1/gradients/output_layer_1/Reshape_grad/ShapeShape hidden_layer_0_1/rnn/transpose_1*
T0*
out_type0
�
9Optimizer_1/gradients/output_layer_1/Reshape_grad/ReshapeReshapeIOptimizer_1/gradients/output_layer_1/MatMul_grad/tuple/control_dependency7Optimizer_1/gradients/output_layer_1/Reshape_grad/Shape*
T0*
Tshape0
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/transpose_1_grad/InvertPermutationInvertPermutationhidden_layer_0_1/rnn/concat_2*
T0
�
EOptimizer_1/gradients/hidden_layer_0_1/rnn/transpose_1_grad/transpose	Transpose9Optimizer_1/gradients/output_layer_1/Reshape_grad/ReshapeMOptimizer_1/gradients/hidden_layer_0_1/rnn/transpose_1_grad/InvertPermutation*
T0*
Tperm0
�
vOptimizer_1/gradients/hidden_layer_0_1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3 hidden_layer_0_1/rnn/TensorArray!hidden_layer_0_1/rnn/while/Exit_2*3
_class)
'%loc:@hidden_layer_0_1/rnn/TensorArray*!
sourceOptimizer_1/gradients
�
rOptimizer_1/gradients/hidden_layer_0_1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flowIdentity!hidden_layer_0_1/rnn/while/Exit_2w^Optimizer_1/gradients/hidden_layer_0_1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*3
_class)
'%loc:@hidden_layer_0_1/rnn/TensorArray
�
|Optimizer_1/gradients/hidden_layer_0_1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3vOptimizer_1/gradients/hidden_layer_0_1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/TensorArrayGradV3+hidden_layer_0_1/rnn/TensorArrayStack/rangeEOptimizer_1/gradients/hidden_layer_0_1/rnn/transpose_1_grad/transposerOptimizer_1/gradients/hidden_layer_0_1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayGrad/gradient_flow*
T0
Y
 Optimizer_1/gradients/zeros_like	ZerosLike!hidden_layer_0_1/rnn/while/Exit_3*
T0
[
"Optimizer_1/gradients/zeros_like_1	ZerosLike!hidden_layer_0_1/rnn/while/Exit_4*
T0
�
COptimizer_1/gradients/hidden_layer_0_1/rnn/while/Exit_2_grad/b_exitEnter|Optimizer_1/gradients/hidden_layer_0_1/rnn/TensorArrayStack/TensorArrayGatherV3_grad/TensorArrayScatter/TensorArrayScatterV3*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
COptimizer_1/gradients/hidden_layer_0_1/rnn/while/Exit_3_grad/b_exitEnter Optimizer_1/gradients/zeros_like*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
COptimizer_1/gradients/hidden_layer_0_1/rnn/while/Exit_4_grad/b_exitEnter"Optimizer_1/gradients/zeros_like_1*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_2_grad/b_switchMergeCOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Exit_2_grad/b_exitNOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_2_grad_1/NextIteration*
N*
T0
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_3_grad/b_switchMergeCOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Exit_3_grad/b_exitNOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_3_grad_1/NextIteration*
N*
T0
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_4_grad/b_switchMergeCOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Exit_4_grad/b_exitNOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_4_grad_1/NextIteration*
N*
T0
�
DOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_2_grad/SwitchSwitchGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_2_grad/b_switchOptimizer_1/gradients/b_count_2*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_2_grad/b_switch
�
NOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_2_grad/tuple/group_depsNoOpE^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_2_grad/Switch
�
VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_2_grad/tuple/control_dependencyIdentityDOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_2_grad/SwitchO^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_2_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_2_grad/b_switch
�
XOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_2_grad/tuple/control_dependency_1IdentityFOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_2_grad/Switch:1O^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_2_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_2_grad/b_switch
�
DOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/SwitchSwitchGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_3_grad/b_switchOptimizer_1/gradients/b_count_2*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_3_grad/b_switch
�
NOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/tuple/group_depsNoOpE^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/Switch
�
VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/tuple/control_dependencyIdentityDOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/SwitchO^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_3_grad/b_switch
�
XOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/tuple/control_dependency_1IdentityFOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/Switch:1O^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_3_grad/b_switch
�
DOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/SwitchSwitchGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_4_grad/b_switchOptimizer_1/gradients/b_count_2*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_4_grad/b_switch
�
NOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/tuple/group_depsNoOpE^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/Switch
�
VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/tuple/control_dependencyIdentityDOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/SwitchO^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_4_grad/b_switch
�
XOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/tuple/control_dependency_1IdentityFOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/Switch:1O^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_4_grad/b_switch
�
BOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Enter_2_grad/ExitExitVOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_2_grad/tuple/control_dependency*
T0
�
BOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Enter_3_grad/ExitExitVOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/tuple/control_dependency*
T0
�
BOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Enter_4_grad/ExitExitVOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/tuple/control_dependency*
T0
�
bOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_outIdentitymOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPopV2*
T0
�
hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/ConstConst*9
_class/
-+loc:@hidden_layer_0_1/rnn/while/dropout/Mul*
dtype0*
valueB :
���������
�
hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/f_accStackV2hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/Const*9
_class/
-+loc:@hidden_layer_0_1/rnn/while/dropout/Mul*
	elem_type0*

stack_name 
�
hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/EnterEnterhOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
nOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPushV2StackPushV2hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/Enter>hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
mOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPopV2
StackPopV2sOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
sOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPopV2/EnterEnterhOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
iOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/b_syncControlTriggerQ^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPopV2U^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/StackPopV2U^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/StackPopV2w^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2n^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out/StackPopV2c^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2e^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1Q^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/StackPopV2S^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/StackPopV2g^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2U^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/StackPopV2Y^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2[^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2g^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2i^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1e^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2g^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1Y^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2g^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2i^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1U^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2W^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2g^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2i^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1U^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2W^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2e^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2g^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1U^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2
�
gOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrierIdentityXOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_2_grad/tuple/control_dependency_1c^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/flow_out*
T0
�
{Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3�Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EntergOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrier*9
_class/
-+loc:@hidden_layer_0_1/rnn/while/dropout/Mul*!
sourceOptimizer_1/gradients
�
�Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnter hidden_layer_0_1/rnn/TensorArray*
T0*9
_class/
-+loc:@hidden_layer_0_1/rnn/while/dropout/Mul*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
wOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentitygOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrier|^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*
T0*9
_class/
-+loc:@hidden_layer_0_1/rnn/while/dropout/Mul
�
kOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3{Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3vOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2wOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0
�
qOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/ConstConst*8
_class.
,*loc:@hidden_layer_0_1/rnn/while/Identity_1*
dtype0*
valueB :
���������
�
qOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStackV2qOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Const*8
_class.
,*loc:@hidden_layer_0_1/rnn/while/Identity_1*
	elem_type0*

stack_name 
�
qOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/EnterEnterqOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
wOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushV2StackPushV2qOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/Enter%hidden_layer_0_1/rnn/while/Identity_1^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
vOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2
StackPopV2|Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
|Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopV2/EnterEnterqOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
jOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_depsNoOpl^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3h^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrier
�
rOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyIdentitykOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3k^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*~
_classt
rploc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3
�
tOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1IdentitygOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrierk^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/group_deps*
T0*z
_classp
nlloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/write_barrier
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like	ZerosLikeTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/StackPopV2*
T0
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/ConstConst*8
_class.
,*loc:@hidden_layer_0_1/rnn/while/Identity_3*
dtype0*
valueB :
���������
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/f_accStackV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/Const*8
_class.
,*loc:@hidden_layer_0_1/rnn/while/Identity_3*
	elem_type0*

stack_name 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/StackPushV2StackPushV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/Enter%hidden_layer_0_1/rnn/while/Identity_3^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/StackPopV2
StackPopV2ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/StackPopV2/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
EOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/SelectSelectPOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPopV2XOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/tuple/control_dependency_1IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like*
T0
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/ConstConst*:
_class0
.,loc:@hidden_layer_0_1/rnn/while/GreaterEqual*
dtype0*
valueB :
���������
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/f_accStackV2KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/Const*:
_class0
.,loc:@hidden_layer_0_1/rnn/while/GreaterEqual*
	elem_type0
*

stack_name 
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/EnterEnterKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPushV2StackPushV2KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/Enter'hidden_layer_0_1/rnn/while/GreaterEqual^Optimizer_1/gradients/Add*
T0
*
swap_memory( 
�
POptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPopV2
StackPopV2VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0

�
VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPopV2/EnterEnterKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select_1SelectPOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPopV2IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_likeXOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_3_grad/tuple/control_dependency_1*
T0
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/tuple/group_depsNoOpF^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/SelectH^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select_1
�
WOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/tuple/control_dependencyIdentityEOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/SelectP^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select
�
YOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/tuple/control_dependency_1IdentityGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select_1P^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select_1
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like	ZerosLikeTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/StackPopV2*
T0
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/ConstConst*8
_class.
,*loc:@hidden_layer_0_1/rnn/while/Identity_4*
dtype0*
valueB :
���������
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/f_accStackV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/Const*8
_class.
,*loc:@hidden_layer_0_1/rnn/while/Identity_4*
	elem_type0*

stack_name 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/StackPushV2StackPushV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/Enter%hidden_layer_0_1/rnn/while/Identity_4^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/StackPopV2
StackPopV2ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/StackPopV2/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
EOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/SelectSelectPOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPopV2XOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/tuple/control_dependency_1IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like*
T0
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/Select_1SelectPOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPopV2IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_likeXOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Merge_4_grad/tuple/control_dependency_1*
T0
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/tuple/group_depsNoOpF^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/SelectH^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/Select_1
�
WOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/tuple/control_dependencyIdentityEOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/SelectP^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/Select
�
YOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/tuple/control_dependency_1IdentityGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/Select_1P^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/tuple/group_deps*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/Select_1
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/zeros_like	ZerosLikeMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/zeros_like/Enter^Optimizer_1/gradients/Sub*
T0
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/zeros_like/EnterEnterhidden_layer_0_1/rnn/zeros*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
COptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/SelectSelectPOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPopV2rOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependencyGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/zeros_like*
T0
�
EOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/Select_1SelectPOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select/StackPopV2GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/zeros_likerOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency*
T0
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/tuple/group_depsNoOpD^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/SelectF^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/Select_1
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/tuple/control_dependencyIdentityCOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/SelectN^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/tuple/group_deps*
T0*V
_classL
JHloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/Select
�
WOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/tuple/control_dependency_1IdentityEOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/Select_1N^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/tuple/group_deps*
T0*X
_classN
LJloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/Select_1
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/ShapeShape*hidden_layer_0_1/rnn/while/dropout/RealDiv*
T0*
out_type0
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Shape_1Shape'hidden_layer_0_1/rnn/while/dropout/Cast*
T0*
out_type0
�
WOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgsBroadcastGradientArgsbOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2dOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
]Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/ConstConst*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Shape*
dtype0*
valueB :
���������
�
]Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_accStackV2]Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/Const*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Shape*
	elem_type0*

stack_name 
�
]Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/EnterEnter]Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2]Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/EnterGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Shape^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
bOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter]Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/Const_1Const*\
_classR
PNloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Shape_1*
dtype0*
valueB :
���������
�
_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_acc_1StackV2_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/Const_1*\
_classR
PNloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Shape_1*
	elem_type0*

stack_name 
�
_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/Enter_1Enter_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_acc_1*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
eOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/Enter_1IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Shape_1^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
dOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2jOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
jOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnter_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/f_acc_1*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
EOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/MulMulWOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/tuple/control_dependency_1POptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/StackPopV2*
T0
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/ConstConst*:
_class0
.,loc:@hidden_layer_0_1/rnn/while/dropout/Cast*
dtype0*
valueB :
���������
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/f_accStackV2KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/Const*:
_class0
.,loc:@hidden_layer_0_1/rnn/while/dropout/Cast*
	elem_type0*

stack_name 
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/EnterEnterKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/StackPushV2StackPushV2KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/Enter'hidden_layer_0_1/rnn/while/dropout/Cast^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
POptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/StackPopV2
StackPopV2VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/StackPopV2/EnterEnterKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
EOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/SumSumEOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/MulWOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/ReshapeReshapeEOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/SumbOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1MulROptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/StackPopV2WOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_grad/tuple/control_dependency_1*
T0
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/ConstConst*=
_class3
1/loc:@hidden_layer_0_1/rnn/while/dropout/RealDiv*
dtype0*
valueB :
���������
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/f_accStackV2MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/Const*=
_class3
1/loc:@hidden_layer_0_1/rnn/while/dropout/RealDiv*
	elem_type0*

stack_name 
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/EnterEnterMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/StackPushV2StackPushV2MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/Enter*hidden_layer_0_1/rnn/while/dropout/RealDiv^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
ROptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/StackPopV2
StackPopV2XOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
XOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/StackPopV2/EnterEnterMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Sum_1SumGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Mul_1YOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Reshape_1ReshapeGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Sum_1dOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
ROptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/tuple/group_depsNoOpJ^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/ReshapeL^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Reshape_1
�
ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/tuple/control_dependencyIdentityIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/ReshapeS^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/tuple/group_deps*
T0*\
_classR
PNloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Reshape
�
\Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/tuple/control_dependency_1IdentityKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Reshape_1S^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/Reshape_1
�
NOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_2_grad_1/NextIterationNextIterationtOptimizer_1/gradients/hidden_layer_0_1/rnn/while/TensorArrayWrite/TensorArrayWriteV3_grad/tuple/control_dependency_1*
T0
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/ShapeShape*hidden_layer_0_1/rnn/while/lstm_cell/mul_2*
T0*
out_type0
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Shape_1Const^Optimizer_1/gradients/Sub*
dtype0*
valueB 
�
[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgsBroadcastGradientArgsfOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Shape_1*
T0
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/ConstConst*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Shape*
dtype0*
valueB :
���������
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/f_accStackV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/Const*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Shape*
	elem_type0*

stack_name 
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/EnterEnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
gOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPushV2StackPushV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/EnterKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Shape^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
fOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2
StackPopV2lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2/EnterEnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDivRealDivZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/tuple/control_dependencyXOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2*
T0
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/ConstConst*9
_class/
-+loc:@hidden_layer_0_1/rnn/while/dropout/Sub*
dtype0*
valueB :
���������
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/f_accStackV2SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/Const*9
_class/
-+loc:@hidden_layer_0_1/rnn/while/dropout/Sub*
	elem_type0*

stack_name 
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/EnterEnterSOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
YOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/StackPushV2StackPushV2SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/Enter&hidden_layer_0_1/rnn/while/dropout/Sub^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
XOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2
StackPopV2^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2/EnterEnterSOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/SumSumMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/ReshapeReshapeIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/SumfOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/NegNegTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/StackPopV2*
T0
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/ConstConst*=
_class3
1/loc:@hidden_layer_0_1/rnn/while/lstm_cell/mul_2*
dtype0*
valueB :
���������
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/f_accStackV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/Const*=
_class3
1/loc:@hidden_layer_0_1/rnn/while/lstm_cell/mul_2*
	elem_type0*

stack_name 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/StackPushV2StackPushV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/Enter*hidden_layer_0_1/rnn/while/lstm_cell/mul_2^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/StackPopV2
StackPopV2ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/StackPopV2/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Neg/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv_1RealDivIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/NegXOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2*
T0
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv_2RealDivOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv_1XOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv/StackPopV2*
T0
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/mulMulZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/Mul_grad/tuple/control_dependencyOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/RealDiv_2*
T0
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Sum_1SumIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/mul]Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Reshape_1ReshapeKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Sum_1MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Shape_1*
T0*
Tshape0
�
VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/tuple/group_depsNoOpN^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/ReshapeP^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Reshape_1
�
^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/tuple/control_dependencyIdentityMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/ReshapeW^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/tuple/group_deps*
T0*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Reshape
�
`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/tuple/control_dependency_1IdentityOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Reshape_1W^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/tuple/group_deps*
T0*b
_classX
VTloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/Reshape_1
�
Optimizer_1/gradients/AddN_4AddNYOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/tuple/control_dependency_1^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/dropout/RealDiv_grad/tuple/control_dependency*
N*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/Select_1
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/ShapeShape.hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_2*
T0*
out_type0
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Shape_1Shape+hidden_layer_0_1/rnn/while/lstm_cell/Tanh_1*
T0*
out_type0
�
[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsfOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/ConstConst*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Shape*
dtype0*
valueB :
���������
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_accStackV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Shape*
	elem_type0*

stack_name 
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterEnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
gOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2StackPushV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/EnterKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Shape^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
fOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2
StackPopV2lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2/EnterEnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1Const*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Shape_1*
dtype0*
valueB :
���������
�
cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1StackV2cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Const_1*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Shape_1*
	elem_type0*

stack_name 
�
cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1EntercOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
iOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/Enter_1MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Shape_1^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2nOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
nOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1/EnterEntercOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/f_acc_1*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/MulMulOptimizer_1/gradients/AddN_4TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2*
T0
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/ConstConst*>
_class4
20loc:@hidden_layer_0_1/rnn/while/lstm_cell/Tanh_1*
dtype0*
valueB :
���������
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/f_accStackV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/Const*>
_class4
20loc:@hidden_layer_0_1/rnn/while/lstm_cell/Tanh_1*
	elem_type0*

stack_name 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/StackPushV2StackPushV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/Enter+hidden_layer_0_1/rnn/while/lstm_cell/Tanh_1^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2
StackPopV2ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/SumSumIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/ReshapeReshapeIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/SumfOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1MulVOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2Optimizer_1/gradients/AddN_4*
T0
�
QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/ConstConst*A
_class7
53loc:@hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_2*
dtype0*
valueB :
���������
�
QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_accStackV2QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/Const*A
_class7
53loc:@hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_2*
	elem_type0*

stack_name 
�
QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/EnterEnterQOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
WOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPushV2StackPushV2QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/Enter.hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_2^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2
StackPopV2\Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
\Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2/EnterEnterQOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Sum_1SumKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1]Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Reshape_1ReshapeKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Sum_1hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/tuple/group_depsNoOpN^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/ReshapeP^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Reshape_1
�
^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependencyIdentityMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/ReshapeW^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Reshape
�
`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1IdentityOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Reshape_1W^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/tuple/group_deps*
T0*b
_classX
VTloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Reshape_1
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradSigmoidGradVOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul_1/StackPopV2^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency*
T0
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/Tanh_1_grad/TanhGradTanhGradTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/Mul/StackPopV2`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_2_grad/tuple/control_dependency_1*
T0
�
Optimizer_1/gradients/AddN_5AddNYOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/tuple/control_dependency_1OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/Tanh_1_grad/TanhGrad*
N*
T0*Z
_classP
NLloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select_1
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/ShapeShape(hidden_layer_0_1/rnn/while/lstm_cell/mul*
T0*
out_type0
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Shape_1Shape*hidden_layer_0_1/rnn/while/lstm_cell/mul_1*
T0*
out_type0
�
[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsfOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/ConstConst*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Shape*
dtype0*
valueB :
���������
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_accStackV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Shape*
	elem_type0*

stack_name 
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterEnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
gOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/EnterKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Shape^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
fOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1Const*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Shape_1*
dtype0*
valueB :
���������
�
cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1StackV2cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Const_1*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Shape_1*
	elem_type0*

stack_name 
�
cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1EntercOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
iOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/Enter_1MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Shape_1^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2nOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
nOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEntercOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/f_acc_1*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/SumSumOptimizer_1/gradients/AddN_5[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/ReshapeReshapeIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/SumfOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Sum_1SumOptimizer_1/gradients/AddN_5]Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Reshape_1ReshapeKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Sum_1hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/tuple/group_depsNoOpN^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/ReshapeP^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Reshape_1
�
^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyIdentityMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/ReshapeW^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Reshape
�
`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1IdentityOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Reshape_1W^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/tuple/group_deps*
T0*b
_classX
VTloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/Reshape_1
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/ShapeShape,hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid*
T0*
out_type0
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Shape_1Shape%hidden_layer_0_1/rnn/while/Identity_3*
T0*
out_type0
�
YOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgsBroadcastGradientArgsdOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2fOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/ConstConst*\
_classR
PNloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Shape*
dtype0*
valueB :
���������
�
_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_accStackV2_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const*\
_classR
PNloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Shape*
	elem_type0*

stack_name 
�
_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterEnter_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
eOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2StackPushV2_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/EnterIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Shape^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
dOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2
StackPopV2jOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
jOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2/EnterEnter_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1Const*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Shape_1*
dtype0*
valueB :
���������
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1StackV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Const_1*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Shape_1*
	elem_type0*

stack_name 
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1EnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
gOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/Enter_1KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Shape_1^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
fOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/f_acc_1*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/MulMul^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/tuple/control_dependencyTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/zeros_like/StackPopV2*
T0
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/SumSumGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/MulYOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/ReshapeReshapeGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/SumdOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1MulTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency*
T0
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/ConstConst*?
_class5
31loc:@hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid*
dtype0*
valueB :
���������
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/f_accStackV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/Const*?
_class5
31loc:@hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid*
	elem_type0*

stack_name 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/StackPushV2StackPushV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/Enter,hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2
StackPopV2ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Sum_1SumIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Reshape_1ReshapeIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Sum_1fOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/tuple/group_depsNoOpL^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/ReshapeN^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Reshape_1
�
\Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/tuple/control_dependencyIdentityKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/ReshapeU^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/tuple/group_deps*
T0*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Reshape
�
^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1IdentityMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Reshape_1U^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Reshape_1
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/ShapeShape.hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_1*
T0*
out_type0
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Shape_1Shape)hidden_layer_0_1/rnn/while/lstm_cell/Tanh*
T0*
out_type0
�
[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsfOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/ConstConst*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Shape*
dtype0*
valueB :
���������
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_accStackV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Shape*
	elem_type0*

stack_name 
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterEnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
gOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2StackPushV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/EnterKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Shape^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
fOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2
StackPopV2lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2/EnterEnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1Const*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Shape_1*
dtype0*
valueB :
���������
�
cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1StackV2cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Const_1*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Shape_1*
	elem_type0*

stack_name 
�
cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1EntercOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
iOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2cOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/Enter_1MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Shape_1^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2nOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
nOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1/EnterEntercOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/f_acc_1*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/MulMul`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2*
T0
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/ConstConst*<
_class2
0.loc:@hidden_layer_0_1/rnn/while/lstm_cell/Tanh*
dtype0*
valueB :
���������
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/f_accStackV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/Const*<
_class2
0.loc:@hidden_layer_0_1/rnn/while/lstm_cell/Tanh*
	elem_type0*

stack_name 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/StackPushV2StackPushV2OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/Enter)hidden_layer_0_1/rnn/while/lstm_cell/Tanh^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2
StackPopV2ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2/EnterEnterOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/SumSumIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/ReshapeReshapeIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/SumfOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1MulVOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_1_grad/tuple/control_dependency_1*
T0
�
QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/ConstConst*A
_class7
53loc:@hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_1*
dtype0*
valueB :
���������
�
QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_accStackV2QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/Const*A
_class7
53loc:@hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_1*
	elem_type0*

stack_name 
�
QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/EnterEnterQOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
WOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPushV2StackPushV2QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/Enter.hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_1^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2
StackPopV2\Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
\Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2/EnterEnterQOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Sum_1SumKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1]Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Reshape_1ReshapeKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Sum_1hOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/tuple/group_depsNoOpN^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/ReshapeP^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Reshape_1
�
^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependencyIdentityMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/ReshapeW^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Reshape
�
`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1IdentityOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Reshape_1W^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/tuple/group_deps*
T0*b
_classX
VTloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Reshape_1
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradSigmoidGradTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/Mul_1/StackPopV2\Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/tuple/control_dependency*
T0
�
Optimizer_1/gradients/AddN_6AddNWOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/tuple/control_dependency^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_grad/tuple/control_dependency_1*
N*
T0*X
_classN
LJloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_1_grad/Select
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGradSigmoidGradVOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul_1/StackPopV2^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency*
T0
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/Tanh_grad/TanhGradTanhGradTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/Mul/StackPopV2`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/mul_1_grad/tuple/control_dependency_1*
T0
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/ShapeShape,hidden_layer_0_1/rnn/while/lstm_cell/split:2*
T0*
out_type0
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Shape_1Shape*hidden_layer_0_1/rnn/while/lstm_cell/add/y*
T0*
out_type0
�
YOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgsBroadcastGradientArgsdOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2fOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1*
T0
�
_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/ConstConst*\
_classR
PNloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Shape*
dtype0*
valueB :
���������
�
_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_accStackV2_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Const*\
_classR
PNloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Shape*
	elem_type0*

stack_name 
�
_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterEnter_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
eOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2StackPushV2_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/EnterIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Shape^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
dOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2
StackPopV2jOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
jOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2/EnterEnter_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Const_1Const*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Shape_1*
dtype0*
valueB :
���������
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc_1StackV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Const_1*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Shape_1*
	elem_type0*

stack_name 
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter_1EnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc_1*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
gOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPushV2_1StackPushV2aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/Enter_1KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Shape_1^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
fOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1
StackPopV2lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
lOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1/EnterEnteraOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/f_acc_1*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
GOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/SumSumSOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGradYOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( 
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/ReshapeReshapeGOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/SumdOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2*
T0*
Tshape0
�
IOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Sum_1SumSOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_grad/SigmoidGrad[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( 
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Reshape_1ReshapeIOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Sum_1fOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/BroadcastGradientArgs/StackPopV2_1*
T0*
Tshape0
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/tuple/group_depsNoOpL^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/ReshapeN^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Reshape_1
�
\Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/tuple/control_dependencyIdentityKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/ReshapeU^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*^
_classT
RPloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Reshape
�
^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/tuple/control_dependency_1IdentityMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Reshape_1U^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/tuple/group_deps*
T0*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/Reshape_1
�
NOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_3_grad_1/NextIterationNextIterationOptimizer_1/gradients/AddN_6*
T0
�
LOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/split_grad/concatConcatV2UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_1_grad/SigmoidGradMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/Tanh_grad/TanhGrad\Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/add_grad/tuple/control_dependencyUOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/Sigmoid_2_grad/SigmoidGradROptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/split_grad/concat/Const*
N*
T0*

Tidx0
�
ROptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/split_grad/concat/ConstConst^Optimizer_1/gradients/Sub*
dtype0*
value	B :
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradBiasAddGradLOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/split_grad/concat*
T0*
data_formatNHWC
�
XOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_depsNoOpT^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradM^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/split_grad/concat
�
`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencyIdentityLOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/split_grad/concatY^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*_
_classU
SQloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/split_grad/concat
�
bOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1IdentitySOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGradY^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/tuple/group_deps*
T0*f
_class\
ZXloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/BiasAddGrad
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMulMatMul`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependencySOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul/Enter*
T0*
transpose_a( *
transpose_b(
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul/EnterEnter(hidden_layer_0/rnn/lstm_cell/kernel/read*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1MatMulZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(*
transpose_b( 
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/ConstConst*>
_class4
20loc:@hidden_layer_0_1/rnn/while/lstm_cell/concat*
dtype0*
valueB :
���������
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_accStackV2UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Const*>
_class4
20loc:@hidden_layer_0_1/rnn/while/lstm_cell/concat*
	elem_type0*

stack_name 
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/EnterEnterUOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPushV2StackPushV2UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/Enter+hidden_layer_0_1/rnn/while/lstm_cell/concat^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2
StackPopV2`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
`Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/StackPopV2/EnterEnterUOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
WOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/tuple/group_depsNoOpN^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMulP^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1
�
_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyIdentityMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMulX^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*`
_classV
TRloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1IdentityOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1X^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/tuple/group_deps*
T0*b
_classX
VTloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/MatMul_1
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_accConst*
dtype0*
valueB�*    
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1EnterSOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2MergeUOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_1[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIteration*
N*
T0
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/SwitchSwitchUOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_2Optimizer_1/gradients/b_count_2*
T0
�
QOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/AddAddV2VOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch:1bOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd_grad/tuple/control_dependency_1*
T0
�
[Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/NextIterationNextIterationQOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/Add*
T0
�
UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3ExitTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/Switch*
T0
�
LOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ConstConst^Optimizer_1/gradients/Sub*
dtype0*
value	B :
�
KOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/RankConst^Optimizer_1/gradients/Sub*
dtype0*
value	B :
�
JOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/modFloorModLOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ConstKOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/Rank*
T0
�
LOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeShape,hidden_layer_0_1/rnn/while/TensorArrayReadV3*
T0*
out_type0
�
MOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeNShapeNXOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/zeros_like/StackPopV2*
N*
T0*
out_type0
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/ConstConst*?
_class5
31loc:@hidden_layer_0_1/rnn/while/TensorArrayReadV3*
dtype0*
valueB :
���������
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/f_accStackV2SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/Const*?
_class5
31loc:@hidden_layer_0_1/rnn/while/TensorArrayReadV3*
	elem_type0*

stack_name 
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/EnterEnterSOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*8

frame_name*(hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
YOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/StackPushV2StackPushV2SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/Enter,hidden_layer_0_1/rnn/while/TensorArrayReadV3^Optimizer_1/gradients/Add*
T0*
swap_memory( 
�
XOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2
StackPopV2^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/Enter^Optimizer_1/gradients/Sub*
	elem_type0
�
^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/StackPopV2/EnterEnterSOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN/f_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant(*
parallel_iterations 
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ConcatOffsetConcatOffsetJOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/modMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeNOOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN:1*
N
�
LOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/SliceSlice_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencySOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ConcatOffsetMOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN*
Index0*
T0
�
NOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/Slice_1Slice_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependencyUOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ConcatOffset:1OOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/ShapeN:1*
Index0*
T0
�
WOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/tuple/group_depsNoOpM^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/SliceO^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/Slice_1
�
_Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/tuple/control_dependencyIdentityLOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/SliceX^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/tuple/group_deps*
T0*_
_classU
SQloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/Slice
�
aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1IdentityNOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/Slice_1X^Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/tuple/group_deps*
T0*a
_classW
USloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/Slice_1
�
ROptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/b_accConst*
dtype0*
valueB
��*    
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1EnterROptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc*
T0*N

frame_name@>Optimizer_1/gradients/hidden_layer_0_1/rnn/while/while_context*
is_constant( *
parallel_iterations 
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2MergeTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_1ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/NextIteration*
N*
T0
�
SOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/SwitchSwitchTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_2Optimizer_1/gradients/b_count_2*
T0
�
POptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/AddAddV2UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/Switch:1aOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul_grad/tuple/control_dependency_1*
T0
�
ZOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/NextIterationNextIterationPOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/Add*
T0
�
TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3ExitSOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/Switch*
T0
�
Optimizer_1/gradients/AddN_7AddNWOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/tuple/control_dependencyaOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/concat_grad/tuple/control_dependency_1*
N*
T0*X
_classN
LJloc:@Optimizer_1/gradients/hidden_layer_0_1/rnn/while/Select_2_grad/Select
�
NOptimizer_1/gradients/hidden_layer_0_1/rnn/while/Switch_4_grad_1/NextIterationNextIterationOptimizer_1/gradients/AddN_7*
T0
�
Optimizer_1/clip_by_norm/mulMulTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3TOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3*
T0
S
Optimizer_1/clip_by_norm/ConstConst*
dtype0*
valueB"       
�
Optimizer_1/clip_by_norm/SumSumOptimizer_1/clip_by_norm/mulOptimizer_1/clip_by_norm/Const*
T0*

Tidx0*
	keep_dims(
O
"Optimizer_1/clip_by_norm/Greater/yConst*
dtype0*
valueB
 *    
v
 Optimizer_1/clip_by_norm/GreaterGreaterOptimizer_1/clip_by_norm/Sum"Optimizer_1/clip_by_norm/Greater/y*
T0
]
(Optimizer_1/clip_by_norm/ones_like/ShapeConst*
dtype0*
valueB"      
U
(Optimizer_1/clip_by_norm/ones_like/ConstConst*
dtype0*
valueB
 *  �?
�
"Optimizer_1/clip_by_norm/ones_likeFill(Optimizer_1/clip_by_norm/ones_like/Shape(Optimizer_1/clip_by_norm/ones_like/Const*
T0*

index_type0
�
Optimizer_1/clip_by_norm/SelectSelect Optimizer_1/clip_by_norm/GreaterOptimizer_1/clip_by_norm/Sum"Optimizer_1/clip_by_norm/ones_like*
T0
O
Optimizer_1/clip_by_norm/SqrtSqrtOptimizer_1/clip_by_norm/Select*
T0
�
!Optimizer_1/clip_by_norm/Select_1Select Optimizer_1/clip_by_norm/GreaterOptimizer_1/clip_by_norm/SqrtOptimizer_1/clip_by_norm/Sum*
T0
M
 Optimizer_1/clip_by_norm/mul_1/yConst*
dtype0*
valueB
 *  �@
�
Optimizer_1/clip_by_norm/mul_1MulTOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/MatMul/Enter_grad/b_acc_3 Optimizer_1/clip_by_norm/mul_1/y*
T0
O
"Optimizer_1/clip_by_norm/Maximum/yConst*
dtype0*
valueB
 *  �@
{
 Optimizer_1/clip_by_norm/MaximumMaximum!Optimizer_1/clip_by_norm/Select_1"Optimizer_1/clip_by_norm/Maximum/y*
T0
v
 Optimizer_1/clip_by_norm/truedivRealDivOptimizer_1/clip_by_norm/mul_1 Optimizer_1/clip_by_norm/Maximum*
T0
O
Optimizer_1/clip_by_normIdentity Optimizer_1/clip_by_norm/truediv*
T0
�
Optimizer_1/clip_by_norm_1/mulMulUOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3UOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3*
T0
N
 Optimizer_1/clip_by_norm_1/ConstConst*
dtype0*
valueB: 
�
Optimizer_1/clip_by_norm_1/SumSumOptimizer_1/clip_by_norm_1/mul Optimizer_1/clip_by_norm_1/Const*
T0*

Tidx0*
	keep_dims(
Q
$Optimizer_1/clip_by_norm_1/Greater/yConst*
dtype0*
valueB
 *    
|
"Optimizer_1/clip_by_norm_1/GreaterGreaterOptimizer_1/clip_by_norm_1/Sum$Optimizer_1/clip_by_norm_1/Greater/y*
T0
X
*Optimizer_1/clip_by_norm_1/ones_like/ShapeConst*
dtype0*
valueB:
W
*Optimizer_1/clip_by_norm_1/ones_like/ConstConst*
dtype0*
valueB
 *  �?
�
$Optimizer_1/clip_by_norm_1/ones_likeFill*Optimizer_1/clip_by_norm_1/ones_like/Shape*Optimizer_1/clip_by_norm_1/ones_like/Const*
T0*

index_type0
�
!Optimizer_1/clip_by_norm_1/SelectSelect"Optimizer_1/clip_by_norm_1/GreaterOptimizer_1/clip_by_norm_1/Sum$Optimizer_1/clip_by_norm_1/ones_like*
T0
S
Optimizer_1/clip_by_norm_1/SqrtSqrt!Optimizer_1/clip_by_norm_1/Select*
T0
�
#Optimizer_1/clip_by_norm_1/Select_1Select"Optimizer_1/clip_by_norm_1/GreaterOptimizer_1/clip_by_norm_1/SqrtOptimizer_1/clip_by_norm_1/Sum*
T0
O
"Optimizer_1/clip_by_norm_1/mul_1/yConst*
dtype0*
valueB
 *  �@
�
 Optimizer_1/clip_by_norm_1/mul_1MulUOptimizer_1/gradients/hidden_layer_0_1/rnn/while/lstm_cell/BiasAdd/Enter_grad/b_acc_3"Optimizer_1/clip_by_norm_1/mul_1/y*
T0
Q
$Optimizer_1/clip_by_norm_1/Maximum/yConst*
dtype0*
valueB
 *  �@
�
"Optimizer_1/clip_by_norm_1/MaximumMaximum#Optimizer_1/clip_by_norm_1/Select_1$Optimizer_1/clip_by_norm_1/Maximum/y*
T0
|
"Optimizer_1/clip_by_norm_1/truedivRealDiv Optimizer_1/clip_by_norm_1/mul_1"Optimizer_1/clip_by_norm_1/Maximum*
T0
S
Optimizer_1/clip_by_norm_1Identity"Optimizer_1/clip_by_norm_1/truediv*
T0
�
Optimizer_1/clip_by_norm_2/mulMulKOptimizer_1/gradients/output_layer_1/MatMul_grad/tuple/control_dependency_1KOptimizer_1/gradients/output_layer_1/MatMul_grad/tuple/control_dependency_1*
T0
U
 Optimizer_1/clip_by_norm_2/ConstConst*
dtype0*
valueB"       
�
Optimizer_1/clip_by_norm_2/SumSumOptimizer_1/clip_by_norm_2/mul Optimizer_1/clip_by_norm_2/Const*
T0*

Tidx0*
	keep_dims(
Q
$Optimizer_1/clip_by_norm_2/Greater/yConst*
dtype0*
valueB
 *    
|
"Optimizer_1/clip_by_norm_2/GreaterGreaterOptimizer_1/clip_by_norm_2/Sum$Optimizer_1/clip_by_norm_2/Greater/y*
T0
_
*Optimizer_1/clip_by_norm_2/ones_like/ShapeConst*
dtype0*
valueB"      
W
*Optimizer_1/clip_by_norm_2/ones_like/ConstConst*
dtype0*
valueB
 *  �?
�
$Optimizer_1/clip_by_norm_2/ones_likeFill*Optimizer_1/clip_by_norm_2/ones_like/Shape*Optimizer_1/clip_by_norm_2/ones_like/Const*
T0*

index_type0
�
!Optimizer_1/clip_by_norm_2/SelectSelect"Optimizer_1/clip_by_norm_2/GreaterOptimizer_1/clip_by_norm_2/Sum$Optimizer_1/clip_by_norm_2/ones_like*
T0
S
Optimizer_1/clip_by_norm_2/SqrtSqrt!Optimizer_1/clip_by_norm_2/Select*
T0
�
#Optimizer_1/clip_by_norm_2/Select_1Select"Optimizer_1/clip_by_norm_2/GreaterOptimizer_1/clip_by_norm_2/SqrtOptimizer_1/clip_by_norm_2/Sum*
T0
O
"Optimizer_1/clip_by_norm_2/mul_1/yConst*
dtype0*
valueB
 *  �@
�
 Optimizer_1/clip_by_norm_2/mul_1MulKOptimizer_1/gradients/output_layer_1/MatMul_grad/tuple/control_dependency_1"Optimizer_1/clip_by_norm_2/mul_1/y*
T0
Q
$Optimizer_1/clip_by_norm_2/Maximum/yConst*
dtype0*
valueB
 *  �@
�
"Optimizer_1/clip_by_norm_2/MaximumMaximum#Optimizer_1/clip_by_norm_2/Select_1$Optimizer_1/clip_by_norm_2/Maximum/y*
T0
|
"Optimizer_1/clip_by_norm_2/truedivRealDiv Optimizer_1/clip_by_norm_2/mul_1"Optimizer_1/clip_by_norm_2/Maximum*
T0
S
Optimizer_1/clip_by_norm_2Identity"Optimizer_1/clip_by_norm_2/truediv*
T0
�
Optimizer_1/clip_by_norm_3/mulMulHOptimizer_1/gradients/output_layer_1/add_grad/tuple/control_dependency_1HOptimizer_1/gradients/output_layer_1/add_grad/tuple/control_dependency_1*
T0
N
 Optimizer_1/clip_by_norm_3/ConstConst*
dtype0*
valueB: 
�
Optimizer_1/clip_by_norm_3/SumSumOptimizer_1/clip_by_norm_3/mul Optimizer_1/clip_by_norm_3/Const*
T0*

Tidx0*
	keep_dims(
Q
$Optimizer_1/clip_by_norm_3/Greater/yConst*
dtype0*
valueB
 *    
|
"Optimizer_1/clip_by_norm_3/GreaterGreaterOptimizer_1/clip_by_norm_3/Sum$Optimizer_1/clip_by_norm_3/Greater/y*
T0
X
*Optimizer_1/clip_by_norm_3/ones_like/ShapeConst*
dtype0*
valueB:
W
*Optimizer_1/clip_by_norm_3/ones_like/ConstConst*
dtype0*
valueB
 *  �?
�
$Optimizer_1/clip_by_norm_3/ones_likeFill*Optimizer_1/clip_by_norm_3/ones_like/Shape*Optimizer_1/clip_by_norm_3/ones_like/Const*
T0*

index_type0
�
!Optimizer_1/clip_by_norm_3/SelectSelect"Optimizer_1/clip_by_norm_3/GreaterOptimizer_1/clip_by_norm_3/Sum$Optimizer_1/clip_by_norm_3/ones_like*
T0
S
Optimizer_1/clip_by_norm_3/SqrtSqrt!Optimizer_1/clip_by_norm_3/Select*
T0
�
#Optimizer_1/clip_by_norm_3/Select_1Select"Optimizer_1/clip_by_norm_3/GreaterOptimizer_1/clip_by_norm_3/SqrtOptimizer_1/clip_by_norm_3/Sum*
T0
O
"Optimizer_1/clip_by_norm_3/mul_1/yConst*
dtype0*
valueB
 *  �@
�
 Optimizer_1/clip_by_norm_3/mul_1MulHOptimizer_1/gradients/output_layer_1/add_grad/tuple/control_dependency_1"Optimizer_1/clip_by_norm_3/mul_1/y*
T0
Q
$Optimizer_1/clip_by_norm_3/Maximum/yConst*
dtype0*
valueB
 *  �@
�
"Optimizer_1/clip_by_norm_3/MaximumMaximum#Optimizer_1/clip_by_norm_3/Select_1$Optimizer_1/clip_by_norm_3/Maximum/y*
T0
|
"Optimizer_1/clip_by_norm_3/truedivRealDiv Optimizer_1/clip_by_norm_3/mul_1"Optimizer_1/clip_by_norm_3/Maximum*
T0
S
Optimizer_1/clip_by_norm_3Identity"Optimizer_1/clip_by_norm_3/truediv*
T0
�
%Optimizer_1/beta1_power/initial_valueConst*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
dtype0*
valueB
 *fff?
�
Optimizer_1/beta1_power
VariableV2*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
	container *
dtype0*
shape: *
shared_name 
�
Optimizer_1/beta1_power/AssignAssignOptimizer_1/beta1_power%Optimizer_1/beta1_power/initial_value*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
Optimizer_1/beta1_power/readIdentityOptimizer_1/beta1_power*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias
�
%Optimizer_1/beta2_power/initial_valueConst*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
dtype0*
valueB
 *w�?
�
Optimizer_1/beta2_power
VariableV2*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
	container *
dtype0*
shape: *
shared_name 
�
Optimizer_1/beta2_power/AssignAssignOptimizer_1/beta2_power%Optimizer_1/beta2_power/initial_value*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
Optimizer_1/beta2_power/readIdentityOptimizer_1/beta2_power*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias
K
Optimizer_1/Adam/learning_rateConst*
dtype0*
valueB
 *j.�;
C
Optimizer_1/Adam/beta1Const*
dtype0*
valueB
 *fff?
C
Optimizer_1/Adam/beta2Const*
dtype0*
valueB
 *w�?
E
Optimizer_1/Adam/epsilonConst*
dtype0*
valueB
 *w�+2
�
EOptimizer_1/Adam/update_hidden_layer_0/rnn/lstm_cell/kernel/ApplyAdam	ApplyAdam#hidden_layer_0/rnn/lstm_cell/kernel2Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1Optimizer_1/beta1_power/readOptimizer_1/beta2_power/readOptimizer_1/Adam/learning_rateOptimizer_1/Adam/beta1Optimizer_1/Adam/beta2Optimizer_1/Adam/epsilonOptimizer_1/clip_by_norm*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
use_locking( *
use_nesterov( 
�
COptimizer_1/Adam/update_hidden_layer_0/rnn/lstm_cell/bias/ApplyAdam	ApplyAdam!hidden_layer_0/rnn/lstm_cell/bias0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1Optimizer_1/beta1_power/readOptimizer_1/beta2_power/readOptimizer_1/Adam/learning_rateOptimizer_1/Adam/beta1Optimizer_1/Adam/beta2Optimizer_1/Adam/epsilonOptimizer_1/clip_by_norm_1*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking( *
use_nesterov( 
�
6Optimizer_1/Adam/update_output_layer/weights/ApplyAdam	ApplyAdamoutput_layer/weights#Optimizer/output_layer/weights/Adam%Optimizer/output_layer/weights/Adam_1Optimizer_1/beta1_power/readOptimizer_1/beta2_power/readOptimizer_1/Adam/learning_rateOptimizer_1/Adam/beta1Optimizer_1/Adam/beta2Optimizer_1/Adam/epsilonOptimizer_1/clip_by_norm_2*
T0*'
_class
loc:@output_layer/weights*
use_locking( *
use_nesterov( 
�
5Optimizer_1/Adam/update_output_layer/biases/ApplyAdam	ApplyAdamoutput_layer/biases"Optimizer/output_layer/biases/Adam$Optimizer/output_layer/biases/Adam_1Optimizer_1/beta1_power/readOptimizer_1/beta2_power/readOptimizer_1/Adam/learning_rateOptimizer_1/Adam/beta1Optimizer_1/Adam/beta2Optimizer_1/Adam/epsilonOptimizer_1/clip_by_norm_3*
T0*&
_class
loc:@output_layer/biases*
use_locking( *
use_nesterov( 
�
Optimizer_1/Adam/mulMulOptimizer_1/beta1_power/readOptimizer_1/Adam/beta1D^Optimizer_1/Adam/update_hidden_layer_0/rnn/lstm_cell/bias/ApplyAdamF^Optimizer_1/Adam/update_hidden_layer_0/rnn/lstm_cell/kernel/ApplyAdam6^Optimizer_1/Adam/update_output_layer/biases/ApplyAdam7^Optimizer_1/Adam/update_output_layer/weights/ApplyAdam*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias
�
Optimizer_1/Adam/AssignAssignOptimizer_1/beta1_powerOptimizer_1/Adam/mul*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking( *
validate_shape(
�
Optimizer_1/Adam/mul_1MulOptimizer_1/beta2_power/readOptimizer_1/Adam/beta2D^Optimizer_1/Adam/update_hidden_layer_0/rnn/lstm_cell/bias/ApplyAdamF^Optimizer_1/Adam/update_hidden_layer_0/rnn/lstm_cell/kernel/ApplyAdam6^Optimizer_1/Adam/update_output_layer/biases/ApplyAdam7^Optimizer_1/Adam/update_output_layer/weights/ApplyAdam*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias
�
Optimizer_1/Adam/Assign_1AssignOptimizer_1/beta2_powerOptimizer_1/Adam/mul_1*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking( *
validate_shape(
�
Optimizer_1/AdamNoOp^Optimizer_1/Adam/Assign^Optimizer_1/Adam/Assign_1D^Optimizer_1/Adam/update_hidden_layer_0/rnn/lstm_cell/bias/ApplyAdamF^Optimizer_1/Adam/update_hidden_layer_0/rnn/lstm_cell/kernel/ApplyAdam6^Optimizer_1/Adam/update_output_layer/biases/ApplyAdam7^Optimizer_1/Adam/update_output_layer/weights/ApplyAdam
�
init_1NoOp^Optimizer/beta1_power/Assign^Optimizer/beta2_power/Assign8^Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam/Assign:^Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1/Assign:^Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam/Assign<^Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1/Assign*^Optimizer/output_layer/biases/Adam/Assign,^Optimizer/output_layer/biases/Adam_1/Assign+^Optimizer/output_layer/weights/Adam/Assign-^Optimizer/output_layer/weights/Adam_1/Assign^Optimizer_1/beta1_power/Assign^Optimizer_1/beta2_power/Assign)^hidden_layer_0/rnn/lstm_cell/bias/Assign+^hidden_layer_0/rnn/lstm_cell/kernel/Assign^output_layer/biases/Assign^output_layer/weights/Assign
A
save/filename/inputConst*
dtype0*
valueB Bmodel
V
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
shape: 
M

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: 
�
save/SaveV2/tensor_namesConst*
dtype0*�
value�B�BOptimizer/beta1_powerBOptimizer/beta2_powerB0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/AdamB2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1B2Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/AdamB4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1B"Optimizer/output_layer/biases/AdamB$Optimizer/output_layer/biases/Adam_1B#Optimizer/output_layer/weights/AdamB%Optimizer/output_layer/weights/Adam_1BOptimizer_1/beta1_powerBOptimizer_1/beta2_powerB!hidden_layer_0/rnn/lstm_cell/biasB#hidden_layer_0/rnn/lstm_cell/kernelBoutput_layer/biasesBoutput_layer/weights
g
save/SaveV2/shape_and_slicesConst*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesOptimizer/beta1_powerOptimizer/beta2_power0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_12Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1"Optimizer/output_layer/biases/Adam$Optimizer/output_layer/biases/Adam_1#Optimizer/output_layer/weights/Adam%Optimizer/output_layer/weights/Adam_1Optimizer_1/beta1_powerOptimizer_1/beta2_power!hidden_layer_0/rnn/lstm_cell/bias#hidden_layer_0/rnn/lstm_cell/kerneloutput_layer/biasesoutput_layer/weights*
dtypes
2
e
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*�
value�B�BOptimizer/beta1_powerBOptimizer/beta2_powerB0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/AdamB2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1B2Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/AdamB4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1B"Optimizer/output_layer/biases/AdamB$Optimizer/output_layer/biases/Adam_1B#Optimizer/output_layer/weights/AdamB%Optimizer/output_layer/weights/Adam_1BOptimizer_1/beta1_powerBOptimizer_1/beta2_powerB!hidden_layer_0/rnn/lstm_cell/biasB#hidden_layer_0/rnn/lstm_cell/kernelBoutput_layer/biasesBoutput_layer/weights
y
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save/AssignAssignOptimizer/beta1_powersave/RestoreV2*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save/Assign_1AssignOptimizer/beta2_powersave/RestoreV2:1*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save/Assign_2Assign0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adamsave/RestoreV2:2*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save/Assign_3Assign2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1save/RestoreV2:3*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save/Assign_4Assign2Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adamsave/RestoreV2:4*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
use_locking(*
validate_shape(
�
save/Assign_5Assign4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1save/RestoreV2:5*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
use_locking(*
validate_shape(
�
save/Assign_6Assign"Optimizer/output_layer/biases/Adamsave/RestoreV2:6*
T0*&
_class
loc:@output_layer/biases*
use_locking(*
validate_shape(
�
save/Assign_7Assign$Optimizer/output_layer/biases/Adam_1save/RestoreV2:7*
T0*&
_class
loc:@output_layer/biases*
use_locking(*
validate_shape(
�
save/Assign_8Assign#Optimizer/output_layer/weights/Adamsave/RestoreV2:8*
T0*'
_class
loc:@output_layer/weights*
use_locking(*
validate_shape(
�
save/Assign_9Assign%Optimizer/output_layer/weights/Adam_1save/RestoreV2:9*
T0*'
_class
loc:@output_layer/weights*
use_locking(*
validate_shape(
�
save/Assign_10AssignOptimizer_1/beta1_powersave/RestoreV2:10*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save/Assign_11AssignOptimizer_1/beta2_powersave/RestoreV2:11*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save/Assign_12Assign!hidden_layer_0/rnn/lstm_cell/biassave/RestoreV2:12*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save/Assign_13Assign#hidden_layer_0/rnn/lstm_cell/kernelsave/RestoreV2:13*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
use_locking(*
validate_shape(
�
save/Assign_14Assignoutput_layer/biasessave/RestoreV2:14*
T0*&
_class
loc:@output_layer/biases*
use_locking(*
validate_shape(
�
save/Assign_15Assignoutput_layer/weightssave/RestoreV2:15*
T0*'
_class
loc:@output_layer/weights*
use_locking(*
validate_shape(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
C
save_1/filename/inputConst*
dtype0*
valueB Bmodel
Z
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
shape: 
Q
save_1/ConstPlaceholderWithDefaultsave_1/filename*
dtype0*
shape: 
�
save_1/SaveV2/tensor_namesConst*
dtype0*�
value�B�BOptimizer/beta1_powerBOptimizer/beta2_powerB0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/AdamB2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1B2Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/AdamB4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1B"Optimizer/output_layer/biases/AdamB$Optimizer/output_layer/biases/Adam_1B#Optimizer/output_layer/weights/AdamB%Optimizer/output_layer/weights/Adam_1BOptimizer_1/beta1_powerBOptimizer_1/beta2_powerB!hidden_layer_0/rnn/lstm_cell/biasB#hidden_layer_0/rnn/lstm_cell/kernelBoutput_layer/biasesBoutput_layer/weights
i
save_1/SaveV2/shape_and_slicesConst*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 
�
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesOptimizer/beta1_powerOptimizer/beta2_power0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_12Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1"Optimizer/output_layer/biases/Adam$Optimizer/output_layer/biases/Adam_1#Optimizer/output_layer/weights/Adam%Optimizer/output_layer/weights/Adam_1Optimizer_1/beta1_powerOptimizer_1/beta2_power!hidden_layer_0/rnn/lstm_cell/bias#hidden_layer_0/rnn/lstm_cell/kerneloutput_layer/biasesoutput_layer/weights*
dtypes
2
m
save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
T0*
_class
loc:@save_1/Const
�
save_1/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*�
value�B�BOptimizer/beta1_powerBOptimizer/beta2_powerB0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/AdamB2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1B2Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/AdamB4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1B"Optimizer/output_layer/biases/AdamB$Optimizer/output_layer/biases/Adam_1B#Optimizer/output_layer/weights/AdamB%Optimizer/output_layer/weights/Adam_1BOptimizer_1/beta1_powerBOptimizer_1/beta2_powerB!hidden_layer_0/rnn/lstm_cell/biasB#hidden_layer_0/rnn/lstm_cell/kernelBoutput_layer/biasesBoutput_layer/weights
{
!save_1/RestoreV2/shape_and_slicesConst"/device:CPU:0*
dtype0*3
value*B(B B B B B B B B B B B B B B B B 
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices"/device:CPU:0*
dtypes
2
�
save_1/AssignAssignOptimizer/beta1_powersave_1/RestoreV2*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save_1/Assign_1AssignOptimizer/beta2_powersave_1/RestoreV2:1*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save_1/Assign_2Assign0Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adamsave_1/RestoreV2:2*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save_1/Assign_3Assign2Optimizer/hidden_layer_0/rnn/lstm_cell/bias/Adam_1save_1/RestoreV2:3*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save_1/Assign_4Assign2Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adamsave_1/RestoreV2:4*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
use_locking(*
validate_shape(
�
save_1/Assign_5Assign4Optimizer/hidden_layer_0/rnn/lstm_cell/kernel/Adam_1save_1/RestoreV2:5*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
use_locking(*
validate_shape(
�
save_1/Assign_6Assign"Optimizer/output_layer/biases/Adamsave_1/RestoreV2:6*
T0*&
_class
loc:@output_layer/biases*
use_locking(*
validate_shape(
�
save_1/Assign_7Assign$Optimizer/output_layer/biases/Adam_1save_1/RestoreV2:7*
T0*&
_class
loc:@output_layer/biases*
use_locking(*
validate_shape(
�
save_1/Assign_8Assign#Optimizer/output_layer/weights/Adamsave_1/RestoreV2:8*
T0*'
_class
loc:@output_layer/weights*
use_locking(*
validate_shape(
�
save_1/Assign_9Assign%Optimizer/output_layer/weights/Adam_1save_1/RestoreV2:9*
T0*'
_class
loc:@output_layer/weights*
use_locking(*
validate_shape(
�
save_1/Assign_10AssignOptimizer_1/beta1_powersave_1/RestoreV2:10*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save_1/Assign_11AssignOptimizer_1/beta2_powersave_1/RestoreV2:11*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save_1/Assign_12Assign!hidden_layer_0/rnn/lstm_cell/biassave_1/RestoreV2:12*
T0*4
_class*
(&loc:@hidden_layer_0/rnn/lstm_cell/bias*
use_locking(*
validate_shape(
�
save_1/Assign_13Assign#hidden_layer_0/rnn/lstm_cell/kernelsave_1/RestoreV2:13*
T0*6
_class,
*(loc:@hidden_layer_0/rnn/lstm_cell/kernel*
use_locking(*
validate_shape(
�
save_1/Assign_14Assignoutput_layer/biasessave_1/RestoreV2:14*
T0*&
_class
loc:@output_layer/biases*
use_locking(*
validate_shape(
�
save_1/Assign_15Assignoutput_layer/weightssave_1/RestoreV2:15*
T0*'
_class
loc:@output_layer/weights*
use_locking(*
validate_shape(
�
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9"�