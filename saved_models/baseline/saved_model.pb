��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
�
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.12.02v2.12.0-rc1-12-g0db597d0d758��
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
serving_default_input_1Placeholder*+
_output_shapes
:���������x*
dtype0* 
shape:���������x
z
serving_default_input_2Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
PartitionedCallPartitionedCallserving_default_input_1serving_default_input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_5485

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	
signatures*
* 
* 
* 
�

non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
trace_1* 
* 
O

_variables
_iterations
_learning_rate
_update_step_xla*

serving_default* 
* 
* 

0
1*
* 
* 
* 
* 
* 
* 

0*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
8
	variables
	keras_api
	total
	count*
H
	variables
	keras_api
	 total
	!count
"
_fn_kwargs*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

 0
!1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCallStatefulPartitionedCallsaver_filename	iterationlearning_ratetotal_1count_1totalcountConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_5591
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename	iterationlearning_ratetotal_1count_1totalcount*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_5619��
�
Q
'__inference_baseline_layer_call_fn_5469
input_1
input_2
identity�
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_baseline_layer_call_and_return_conditional_losses_5466d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������x:���������:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:T P
+
_output_shapes
:���������x
!
_user_specified_name	input_1
�
S
'__inference_baseline_layer_call_fn_5491
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_baseline_layer_call_and_return_conditional_losses_5466d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������x:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������x
"
_user_specified_name
inputs_0
�&
n
B__inference_baseline_layer_call_and_return_conditional_losses_5531
inputs_0
inputs_1
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceinputs_0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskP
Repeat/repeatsConst*
_output_shapes
: *
dtype0*
value	B :\
Repeat/CastCastRepeat/repeats:output:0*

DstT0*

SrcT0*
_output_shapes
: `
Repeat/ShapeShapestrided_slice:output:0*
T0*
_output_shapes
::��W
Repeat/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB Y
Repeat/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB l
Repeat/ReshapeReshapeRepeat/Cast:y:0Repeat/Reshape/shape_1:output:0*
T0*
_output_shapes
: W
Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Repeat/ExpandDims
ExpandDimsstrided_slice:output:0Repeat/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������Y
Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :�
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0Repeat/Reshape:output:0*
N*
T0*
_output_shapes
:�
Repeat/TileTileRepeat/ExpandDims:output:0Repeat/Tile/multiples:output:0*
T0*+
_output_shapes
:���������d
Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_sliceStridedSliceRepeat/Shape:output:0#Repeat/strided_slice/stack:output:0%Repeat/strided_slice/stack_1:output:0%Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
Repeat/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_1StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_1/stack:output:0'Repeat/strided_slice_1/stack_1:output:0'Repeat/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl

Repeat/mulMulRepeat/Reshape:output:0Repeat/strided_slice_1:output:0*
T0*
_output_shapes
: f
Repeat/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
Repeat/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_2StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_2/stack:output:0'Repeat/strided_slice_2/stack_1:output:0'Repeat/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask\
Repeat/concat/values_1PackRepeat/mul:z:0*
N*
T0*
_output_shapes
:T
Repeat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Repeat/concatConcatV2Repeat/strided_slice:output:0Repeat/concat/values_1:output:0Repeat/strided_slice_2:output:0Repeat/concat/axis:output:0*
N*
T0*
_output_shapes
:{
Repeat/Reshape_1ReshapeRepeat/Tile:output:0Repeat/concat:output:0*
T0*'
_output_shapes
:���������Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������

ExpandDims
ExpandDimsRepeat/Reshape_1:output:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������_
IdentityIdentityExpandDims:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������x:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������x
"
_user_specified_name
inputs_0
�+
I
__inference__wrapped_model_5381
input_1
input_2
identityq
baseline/strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    s
baseline/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           s
baseline/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
baseline/strided_sliceStridedSliceinput_1%baseline/strided_slice/stack:output:0'baseline/strided_slice/stack_1:output:0'baseline/strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskY
baseline/Repeat/repeatsConst*
_output_shapes
: *
dtype0*
value	B :n
baseline/Repeat/CastCast baseline/Repeat/repeats:output:0*

DstT0*

SrcT0*
_output_shapes
: r
baseline/Repeat/ShapeShapebaseline/strided_slice:output:0*
T0*
_output_shapes
::��`
baseline/Repeat/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB b
baseline/Repeat/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB �
baseline/Repeat/ReshapeReshapebaseline/Repeat/Cast:y:0(baseline/Repeat/Reshape/shape_1:output:0*
T0*
_output_shapes
: `
baseline/Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
baseline/Repeat/ExpandDims
ExpandDimsbaseline/strided_slice:output:0'baseline/Repeat/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������b
 baseline/Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :b
 baseline/Repeat/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :�
baseline/Repeat/Tile/multiplesPack)baseline/Repeat/Tile/multiples/0:output:0)baseline/Repeat/Tile/multiples/1:output:0 baseline/Repeat/Reshape:output:0*
N*
T0*
_output_shapes
:�
baseline/Repeat/TileTile#baseline/Repeat/ExpandDims:output:0'baseline/Repeat/Tile/multiples:output:0*
T0*+
_output_shapes
:���������m
#baseline/Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%baseline/Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%baseline/Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
baseline/Repeat/strided_sliceStridedSlicebaseline/Repeat/Shape:output:0,baseline/Repeat/strided_slice/stack:output:0.baseline/Repeat/strided_slice/stack_1:output:0.baseline/Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_masko
%baseline/Repeat/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:q
'baseline/Repeat/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:q
'baseline/Repeat/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
baseline/Repeat/strided_slice_1StridedSlicebaseline/Repeat/Shape:output:0.baseline/Repeat/strided_slice_1/stack:output:00baseline/Repeat/strided_slice_1/stack_1:output:00baseline/Repeat/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask�
baseline/Repeat/mulMul baseline/Repeat/Reshape:output:0(baseline/Repeat/strided_slice_1:output:0*
T0*
_output_shapes
: o
%baseline/Repeat/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:q
'baseline/Repeat/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: q
'baseline/Repeat/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
baseline/Repeat/strided_slice_2StridedSlicebaseline/Repeat/Shape:output:0.baseline/Repeat/strided_slice_2/stack:output:00baseline/Repeat/strided_slice_2/stack_1:output:00baseline/Repeat/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maskn
baseline/Repeat/concat/values_1Packbaseline/Repeat/mul:z:0*
N*
T0*
_output_shapes
:]
baseline/Repeat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
baseline/Repeat/concatConcatV2&baseline/Repeat/strided_slice:output:0(baseline/Repeat/concat/values_1:output:0(baseline/Repeat/strided_slice_2:output:0$baseline/Repeat/concat/axis:output:0*
N*
T0*
_output_shapes
:�
baseline/Repeat/Reshape_1Reshapebaseline/Repeat/Tile:output:0baseline/Repeat/concat:output:0*
T0*'
_output_shapes
:���������b
baseline/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������
baseline/ExpandDims
ExpandDims"baseline/Repeat/Reshape_1:output:0 baseline/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������h
IdentityIdentitybaseline/ExpandDims:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������x:���������:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:T P
+
_output_shapes
:���������x
!
_user_specified_name	input_1
�&
l
B__inference_baseline_layer_call_and_return_conditional_losses_5422
input_1
input_2
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskP
Repeat/repeatsConst*
_output_shapes
: *
dtype0*
value	B :\
Repeat/CastCastRepeat/repeats:output:0*

DstT0*

SrcT0*
_output_shapes
: `
Repeat/ShapeShapestrided_slice:output:0*
T0*
_output_shapes
::��W
Repeat/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB Y
Repeat/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB l
Repeat/ReshapeReshapeRepeat/Cast:y:0Repeat/Reshape/shape_1:output:0*
T0*
_output_shapes
: W
Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Repeat/ExpandDims
ExpandDimsstrided_slice:output:0Repeat/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������Y
Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :�
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0Repeat/Reshape:output:0*
N*
T0*
_output_shapes
:�
Repeat/TileTileRepeat/ExpandDims:output:0Repeat/Tile/multiples:output:0*
T0*+
_output_shapes
:���������d
Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_sliceStridedSliceRepeat/Shape:output:0#Repeat/strided_slice/stack:output:0%Repeat/strided_slice/stack_1:output:0%Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
Repeat/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_1StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_1/stack:output:0'Repeat/strided_slice_1/stack_1:output:0'Repeat/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl

Repeat/mulMulRepeat/Reshape:output:0Repeat/strided_slice_1:output:0*
T0*
_output_shapes
: f
Repeat/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
Repeat/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_2StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_2/stack:output:0'Repeat/strided_slice_2/stack_1:output:0'Repeat/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask\
Repeat/concat/values_1PackRepeat/mul:z:0*
N*
T0*
_output_shapes
:T
Repeat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Repeat/concatConcatV2Repeat/strided_slice:output:0Repeat/concat/values_1:output:0Repeat/strided_slice_2:output:0Repeat/concat/axis:output:0*
N*
T0*
_output_shapes
:{
Repeat/Reshape_1ReshapeRepeat/Tile:output:0Repeat/concat:output:0*
T0*'
_output_shapes
:���������Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������

ExpandDims
ExpandDimsRepeat/Reshape_1:output:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������_
IdentityIdentityExpandDims:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������x:���������:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:T P
+
_output_shapes
:���������x
!
_user_specified_name	input_1
�7
�
__inference__traced_save_5591
file_prefix*
 read_disablecopyonread_iteration:	 0
&read_1_disablecopyonread_learning_rate: *
 read_2_disablecopyonread_total_1: *
 read_3_disablecopyonread_count_1: (
read_4_disablecopyonread_total: (
read_5_disablecopyonread_count: 
savev2_const
identity_13��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: r
Read/DisableCopyOnReadDisableCopyOnRead read_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp read_disablecopyonread_iteration^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	a
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: Y

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0	*
_output_shapes
: z
Read_1/DisableCopyOnReadDisableCopyOnRead&read_1_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp&read_1_disablecopyonread_learning_rate^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_2/DisableCopyOnReadDisableCopyOnRead read_2_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp read_2_disablecopyonread_total_1^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_3/DisableCopyOnReadDisableCopyOnRead read_3_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp read_3_disablecopyonread_count_1^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
: r
Read_4/DisableCopyOnReadDisableCopyOnReadread_4_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOpread_4_disablecopyonread_total^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0e

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: [

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes
: r
Read_5/DisableCopyOnReadDisableCopyOnReadread_5_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOpread_5_disablecopyonread_count^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0f
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH{
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtypes
	2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_12Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_13IdentityIdentity_12:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "#
identity_13Identity_13:output:0*#
_input_shapes
: : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp:

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
 __inference__traced_restore_5619
file_prefix$
assignvariableop_iteration:	 *
 assignvariableop_1_learning_rate: $
assignvariableop_2_total_1: $
assignvariableop_3_count_1: "
assignvariableop_4_total: "
assignvariableop_5_count: 

identity_7��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH~
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_iterationIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp assignvariableop_1_learning_rateIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_total_1Identity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_count_1Identity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_totalIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_countIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_7IdentityIdentity_6:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*"
_acd_function_control_output(*
_output_shapes
 "!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
L
"__inference_signature_wrapper_5485
input_1
input_2
identity�
PartitionedCallPartitionedCallinput_1input_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__wrapped_model_5381d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������x:���������:PL
'
_output_shapes
:���������
!
_user_specified_name	input_2:T P
+
_output_shapes
:���������x
!
_user_specified_name	input_1
�&
l
B__inference_baseline_layer_call_and_return_conditional_losses_5466

inputs
inputs_1
identityh
strided_slice/stackConst*
_output_shapes
:*
dtype0*!
valueB"    ����    j
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*!
valueB"           j
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*!
valueB"         �
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*'
_output_shapes
:���������*

begin_mask*
end_mask*
shrink_axis_maskP
Repeat/repeatsConst*
_output_shapes
: *
dtype0*
value	B :\
Repeat/CastCastRepeat/repeats:output:0*

DstT0*

SrcT0*
_output_shapes
: `
Repeat/ShapeShapestrided_slice:output:0*
T0*
_output_shapes
::��W
Repeat/Reshape/shapeConst*
_output_shapes
: *
dtype0*
valueB Y
Repeat/Reshape/shape_1Const*
_output_shapes
: *
dtype0*
valueB l
Repeat/ReshapeReshapeRepeat/Cast:y:0Repeat/Reshape/shape_1:output:0*
T0*
_output_shapes
: W
Repeat/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :�
Repeat/ExpandDims
ExpandDimsstrided_slice:output:0Repeat/ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������Y
Repeat/Tile/multiples/0Const*
_output_shapes
: *
dtype0*
value	B :Y
Repeat/Tile/multiples/1Const*
_output_shapes
: *
dtype0*
value	B :�
Repeat/Tile/multiplesPack Repeat/Tile/multiples/0:output:0 Repeat/Tile/multiples/1:output:0Repeat/Reshape:output:0*
N*
T0*
_output_shapes
:�
Repeat/TileTileRepeat/ExpandDims:output:0Repeat/Tile/multiples:output:0*
T0*+
_output_shapes
:���������d
Repeat/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
Repeat/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
Repeat/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_sliceStridedSliceRepeat/Shape:output:0#Repeat/strided_slice/stack:output:0%Repeat/strided_slice/stack_1:output:0%Repeat/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskf
Repeat/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_1StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_1/stack:output:0'Repeat/strided_slice_1/stack_1:output:0'Repeat/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl

Repeat/mulMulRepeat/Reshape:output:0Repeat/strided_slice_1:output:0*
T0*
_output_shapes
: f
Repeat/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:h
Repeat/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
Repeat/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Repeat/strided_slice_2StridedSliceRepeat/Shape:output:0%Repeat/strided_slice_2/stack:output:0'Repeat/strided_slice_2/stack_1:output:0'Repeat/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask\
Repeat/concat/values_1PackRepeat/mul:z:0*
N*
T0*
_output_shapes
:T
Repeat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Repeat/concatConcatV2Repeat/strided_slice:output:0Repeat/concat/values_1:output:0Repeat/strided_slice_2:output:0Repeat/concat/axis:output:0*
N*
T0*
_output_shapes
:{
Repeat/Reshape_1ReshapeRepeat/Tile:output:0Repeat/concat:output:0*
T0*'
_output_shapes
:���������Y
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
����������

ExpandDims
ExpandDimsRepeat/Reshape_1:output:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:���������_
IdentityIdentityExpandDims:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:���������x:���������:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������x
 
_user_specified_nameinputs"�
J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
?
input_14
serving_default_input_1:0���������x
;
input_20
serving_default_input_2:0���������8
output_1,
PartitionedCall:0���������tensorflow/serving/predict:�&
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer
	
signatures"
_tf_keras_model
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_12�
'__inference_baseline_layer_call_fn_5469
'__inference_baseline_layer_call_fn_5491�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
trace_0
trace_12�
B__inference_baseline_layer_call_and_return_conditional_losses_5422
B__inference_baseline_layer_call_and_return_conditional_losses_5531�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�B�
__inference__wrapped_model_5381input_1input_2"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
j

_variables
_iterations
_learning_rate
_update_step_xla"
experimentalOptimizer
,
serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_baseline_layer_call_fn_5469input_1input_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_baseline_layer_call_fn_5491inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_baseline_layer_call_and_return_conditional_losses_5422input_1input_2"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_baseline_layer_call_and_return_conditional_losses_5531inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
'
0"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
"__inference_signature_wrapper_5485input_1input_2"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
N
	variables
	keras_api
	total
	count"
_tf_keras_metric
^
	variables
	keras_api
	 total
	!count
"
_fn_kwargs"
_tf_keras_metric
.
0
1"
trackable_list_wrapper
-
	variables"
_generic_user_object
:  (2total
:  (2count
.
 0
!1"
trackable_list_wrapper
-
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
__inference__wrapped_model_5381�\�Y
R�O
M�J
%�"
input_1���������x
!�
input_2���������
� "7�4
2
output_1&�#
output_1����������
B__inference_baseline_layer_call_and_return_conditional_losses_5422�\�Y
R�O
M�J
%�"
input_1���������x
!�
input_2���������
� "0�-
&�#
tensor_0���������
� �
B__inference_baseline_layer_call_and_return_conditional_losses_5531�^�[
T�Q
O�L
&�#
inputs_0���������x
"�
inputs_1���������
� "0�-
&�#
tensor_0���������
� �
'__inference_baseline_layer_call_fn_5469�\�Y
R�O
M�J
%�"
input_1���������x
!�
input_2���������
� "%�"
unknown����������
'__inference_baseline_layer_call_fn_5491�^�[
T�Q
O�L
&�#
inputs_0���������x
"�
inputs_1���������
� "%�"
unknown����������
"__inference_signature_wrapper_5485�m�j
� 
c�`
0
input_1%�"
input_1���������x
,
input_2!�
input_2���������"7�4
2
output_1&�#
output_1���������