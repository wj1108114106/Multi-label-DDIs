from keras.engine.topology import Layer
from keras.layers import Lambda, Dense, Add, BatchNormalization, Multiply, RNN,Reshape,Permute,Softmax,Concatenate
from keras import backend as K
import tensorflow as tf
import keras


class Caps(Layer):
    def __init__(self, num_heads, input_bias=0.0, forget_bias=1.0, qkv_size=96, **kwargs):
        super(Caps, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.input_bias = input_bias
        self.forget_bias = forget_bias
        self.qkv_size = qkv_size
        self.total_size = self.qkv_size * self.num_heads
        # self.mem_slots = mem_slots



    def call(self, inputs):
        key_size=32
        # inputs=BatchNormalization()(inputs)
        memory=Lambda(lambda x: x[:,:1,:])(inputs)
        inputs_reshape=Lambda(lambda x: x[:,1:2,:])(inputs)


        qkv = Dense(self.total_size)(inputs)
        qkv = BatchNormalization()(qkv)
        qkv_reshape = Reshape((2, self.num_heads, self.qkv_size))(qkv)
        qkv_transpose = Permute((2, 1, 3),)(qkv_reshape) ###(?,3,2,96）
        q = Lambda(lambda x: x[:, :, :, :32])(qkv_transpose)    ####(?,3,2,32)
        k = Lambda(lambda x: x[:, :, :, 32:64])(qkv_transpose)  ####(?,3,2,32)
        v = Lambda(lambda x: x[:, :, :, 64:96])(qkv_transpose)  ####(?,3,2,32)
        # q *= key_size ** -0.5  ####(?,3,2,32)
        z = Permute((1, 3, 2))(k) ###(?,3,32,2)
        dot_product=Lambda(lambda x: tf.einsum('bhkl,bhlc->bhkc', x[0], x[1]))([q,z]) ###(?,3,2,2)

        weights = Softmax()(dot_product) ###(?,3,2,2)

        output =Lambda(lambda x: tf.matmul(x[0],x[1]))([weights,v])  ####(?,3,2,32)
        output_transpose = Permute((2, 1, 3))(output)  ###(B,N,H,V)(?,2,3,32)
        attended_memory =  Reshape((2, 96))(output_transpose) ####(?,2,96)
        ########mlp+short up
        next_memory = BatchNormalization()(Add()([inputs, attended_memory]))  ####(?,2,96)
        next_memory_mlp=Dense(120)(next_memory) ###120
        next_memory_mlp=Dense(96)(next_memory_mlp)
        next_memory=BatchNormalization()(Add()([next_memory_mlp,next_memory]))
        next_memory =Lambda(lambda x: x[:, :1, :])(next_memory)            ####(?,1,96)

        ######caculte gate memory self.create_gates(inputs_reshape,memory)
        memory_tanh = Lambda(lambda x: tf.tanh(x))(memory)  ####(?,1,96)
        inputs_tanh = Reshape((96,))(inputs_reshape) ####(?,96)
        gate_inputs = Dense(2)(inputs_tanh)  ####(?,2)

        gate_inputs = Lambda(lambda x: tf.expand_dims(x,axis=1))(gate_inputs)  ####(?,1,2)
        gate_memory = Dense(2)(memory_tanh)  ####(?,1,2)
        gate_add = Add()([gate_memory, gate_inputs])  ####(?,1,2)
        input_gate, forget_gate = Lambda(tf.split, arguments={'axis': 2, 'num_or_size_splits': 2})(gate_add)  ####(?,1,1)    (?,1,1)
        input_gate_sigmoid = Lambda(lambda x: tf.sigmoid(x+self.input_bias))(input_gate)
        forget_gate_sigmoid = Lambda(lambda x: tf.sigmoid(x+self.forget_bias))(forget_gate) ###(?,1,1)
        next_memory_tanh = Lambda(lambda x: tf.tanh(x))(next_memory) ###(?,1,96)
        next_memory_1 = Multiply()([input_gate_sigmoid, next_memory_tanh])  ###(?,1,96)
        next_memory_2 = Multiply()([forget_gate_sigmoid, memory])   ###(?,1,96)
        next_memory = Add()([next_memory_1, next_memory_2])  #####(?,1,96)
        next_memory_output = Reshape((96,))(next_memory) ######(?,96)
        next_memory_output = Lambda(lambda x: tf.expand_dims(x,axis=1))(next_memory_output) ###(?,1,96)

        final_output = Concatenate(axis=1)([next_memory_output, next_memory]) ###(?,1,96)

        return final_output
