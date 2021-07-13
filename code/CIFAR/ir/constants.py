CONV_TYPE = 'Conv'
BN_TYPE = 'BatchNormalization'
SLICE_TYPE = 'Slice'
CONCAT_TYPE = 'Concat'
MAXPOOL_TYPE = 'MaxPool'
AVGPOOL_TYPE = 'AveragePool'
RELU_TYPE = 'Relu'
ADD_TYPE = 'Add'
FC_TYPE = 'Gemm'
RESHAPE_TYPE = 'Reshape'

OP_ALIAS = {
    CONV_TYPE: 'conv',
    BN_TYPE: 'bn',
    SLICE_TYPE: 'split',
    CONCAT_TYPE: 'concat',
    MAXPOOL_TYPE: 'maxpool',
    AVGPOOL_TYPE: 'avgpool',
    RELU_TYPE: 'relu',
    ADD_TYPE: 'add',
    FC_TYPE: 'fc',
    RESHAPE_TYPE: 'reshape',
}