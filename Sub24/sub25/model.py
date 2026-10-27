import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from keras.layers import InputLayer , Input
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split



class AutoEncoder(Model):
  def __init__(self,input_shape):
    self.num_neurons = [40,40,40]
    
    self.enc_dims = 40
    self.decoder_neurons = self.num_neurons[-1]
    self.to_predict = 899
    # self.encoder_layers = []
    # self.decoder_layers = []
    super(AutoEncoder,self).__init__()
    
    
    self.encoder_lstm_1 = layers.LSTM(self.num_neurons[0],
                                          return_sequences=True,
                                          name="encoder_lstm_1",
                                          recurrent_dropout = 0.3) 
    self.encoder_lstm_middle = layers.LSTM(self.num_neurons[1],
                                          return_sequences=True,
                                          name="encoder_lstm_middle")
    self.projector_layer = layers.Dense(self.num_neurons[1],activation="linear",name = "projector")
    
    self.encoder_lstm_2 = layers.LSTM(self.num_neurons[2],
                                          return_sequences=True,
                                          return_state=True,
                                          name="encoder_lstm_2")

    self.decoder_lstm = layers.LSTM(self.decoder_neurons,
                                         return_sequences=True,
                                         return_state=True,
                                         name="decoder_lstm")
    


    self.DenseQ = layers.Dense(self.enc_dims,name = "Query_Dense")
    self.DenseK = layers.Dense(self.enc_dims,name = "Query_Key")
    self.DenseV = layers.Dense(self.enc_dims,name = "Query_Value")

    num_heads = 4
    self.self_attention_layer = layers.MultiHeadAttention(name = "self_attention",num_heads = num_heads,key_dim = self.enc_dims//num_heads)
    self.cross_attention_layer = layers.MultiHeadAttention(name="attention",num_heads = num_heads,key_dim= self.enc_dims//num_heads)
    self.concat_layer = layers.Concatenate(axis=-1, name="concat_attention")
    self.output_dense_layer = layers.Dense(input_shape[1], name="output_dense",activation="linear") 
    self.output_time_dist = layers.TimeDistributed(self.output_dense_layer, name="output_time_distributed") 



    inputs = tf.keras.Input(input_shape,name = "init_input")
    mean_inputs = layers.Lambda(lambda x: tf.math.reduce_mean(x,axis=1,keepdims=True))(inputs)
    x1 = self.encoder_lstm_1(inputs-mean_inputs)
    x = self.encoder_lstm_middle(x1)
    x1_proj = self.projector_layer(x1)
    x = layers.Add(name = "skip_conn")([x1_proj,x])
    lstm_out , h , c = self.encoder_lstm_2(x)
    states = [h,c]

    key = self.DenseK(lstm_out)
    query = self.DenseQ(lstm_out)
    val = self.DenseV(lstm_out)

    attention_context_for_enc = self.self_attention_layer(query, val,key)
    encoder_out = self.concat_layer([attention_context_for_enc, lstm_out])


    self.encoder_model = Model(inputs = inputs,outputs = [encoder_out] + states , name = "ENCODER")

    target = tf.keras.Input(shape = (self.to_predict,input_shape[1]))
    decoder_lstm_outputs, _, _ = self.decoder_lstm(target-mean_inputs,initial_state=states)

    attention_context = self.cross_attention_layer(decoder_lstm_outputs, encoder_out,encoder_out)
    decoder_combined_context = self.concat_layer([attention_context, decoder_lstm_outputs])
    decoder_pred_outputs = self.output_time_dist(decoder_combined_context)


    self.training_model = Model(inputs=[inputs, target],outputs=decoder_pred_outputs)

    #----------------------------------------------------------------------------------------------------
    inf_decoder_input_prev = Input(shape=(1, input_shape[1]), name="inf_decoder_input") # Previous prediction
    inf_decoder_state_h = Input(shape=(self.decoder_neurons,), name="inf_state_h")          # Previous h state
    inf_decoder_state_c = Input(shape=(self.decoder_neurons,), name="inf_state_c")          # Previous c state
    inf_decoder_states_prev = [inf_decoder_state_h, inf_decoder_state_c]                 # Combined previous states
    inf_encoder_outputs = Input(shape=(input_shape[0], self.num_neurons[-1]+self.enc_dims), name="inf_encoder_outputs") # ALL encoder outputs for attention

        # Connect layers for one step using shared layer instances:
    inf_dec_lstm_out, inf_state_h_out, inf_state_c_out = self.decoder_lstm(
        inf_decoder_input_prev, initial_state=inf_decoder_states_prev # Initialize with previous states
    )
    inf_attention_context = self.cross_attention_layer(inf_dec_lstm_out, inf_encoder_outputs,inf_encoder_outputs) # Calculate attention for this step
    inf_combined_context = self.concat_layer([inf_attention_context, inf_dec_lstm_out])    # Combine
    inf_decoder_pred = self.output_time_dist(inf_combined_context)                       # Predict (shape=(batch, 1, features))

        # Define the single-step inference decoder model
    self.inf_decoder_model = Model(
        inputs=[inf_decoder_input_prev, inf_decoder_states_prev, inf_encoder_outputs],
        outputs=[inf_decoder_pred] + [inf_state_h_out, inf_state_c_out], # Output prediction AND next states
        name="inf_decoder"
    )  

    
                            
    

  
  @tf.function(input_signature=[tf.TensorSpec(shape=(1, 100, 32), dtype=tf.float32)])
  def gen_enc_out(self, input_sequence):
     encoder_out, h, c = self.encoder_model(input_sequence) 
     return encoder_out, h, c

  @tf.function(input_signature=[
      tf.TensorSpec(shape=(1, 1, 32), dtype=tf.float32),
      [tf.TensorSpec(shape=(1, 40), dtype=tf.float32), tf.TensorSpec(shape=(1, 40), dtype=tf.float32)],
      tf.TensorSpec(shape=(1, None, 80), dtype=tf.float32)
  ])
  def predict_step(self, target_seq, state, enc_out):
      output_token, h_new, c_new = self.inf_decoder_model([target_seq, state, enc_out])
      return output_token, h_new, c_new
                        

#   def train(self,X_train,y_train):
#     last_known_step = X_train[:, -1:, :] 
#     future_inputs = y_train[:, :-1, :]
#     decoder_input_data = np.concatenate([last_known_step, future_inputs], axis=1)
    
#     # 4. THE TARGET IS JUST THE ORIGINAL Y_TRAIN
#     decoder_target_data = y_train
#     optim = tf.keras.optimizers.Adam(learning_rate=0.01)
#     self.training_model.compile(optimizer = optim,loss= "mse")
#     self.training_model.summary()
#     self.training_model.fit([X_train,decoder_input_data],y_train,batch_size = 32, epochs = 30,verbose = True, validation_split = 0.2)
#     self.training_model.save_weights("C:\CodingWorkspace\Python\WunderRNN\AE_wts.weights.h5")
    
