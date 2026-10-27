import numpy as np
import utils
from model import AutoEncoder
import os
import tensorflow as tf

class PredictionModel:
    """
    Simple model that predicts the next value as a moving average
    of all previous values in the current sequence.
    """

    def __init__(self):
        self.model = AutoEncoder([100,32])
        self.current_seq_ix = None
        self.sequence_history = []
        self.weight_path = os.path.join(os.getcwd(),"AE_wts.weights.h5")
        self.ct = 5

 
    def predict(self, data_point: utils.DataPoint) -> np.ndarray:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.sequence_history = []
            self.prev_state = None
            self.encoder_history = []
            self.skip_first = True
            self.model.training_model.build(input_shape=[(100,32), (899,32)])
            self.model.training_model.load_weights(self.weight_path)
            self.ct=10
        if not data_point.need_prediction:
            self.sequence_history.append(data_point.state.copy())
            return 
        
        elif data_point.need_prediction:
            self.sequence_history.append(data_point.state.copy()) 
            if (self.ct >=7) :
                self.ct = 0
                temp = np.array(self.sequence_history[-100:]).copy()
                temp = np.reshape(temp, shape=(1, 100, 32))
                enc_out, h, c = self.model.gen_enc_out(temp)
                state = [h,c]   
                target_seq = np.reshape(self.sequence_history[-1],shape = (1,1,32))
                output_token, h_new, c_new = self.model.predict_step(target_seq,state,enc_out)
                self.prev_state = [h_new,c_new]
                self.encoder_history = [enc_out,h,c]
                return np.reshape(output_token[0][0],shape = data_point.state.shape)
            else :
                self.ct += 1
                target_seq = np.reshape(self.sequence_history[-1],shape = (1,1,32))
                output_token, h_new, c_new = self.model.predict_step(target_seq,self.prev_state,self.encoder_history[0])
                self.prev_state = [h_new,c_new]
                return np.reshape(output_token[0][0],shape = data_point.state.shape)
               


if __name__ == "__main__":
    # Check existence of test file
    
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    test_file = f"{CURRENT_DIR}/train.parquet"

    # Create and test our model 
    model = PredictionModel()

    # Load data into scorer
    scorer = utils.ScorerStepByStep(test_file)

   # print("Testing simple model with moving average...")
    print(f"Feature dimensionality: {scorer.dim}")
    print(f"Number of rows in dataset: {len(scorer.dataset)}")

    # Evaluate our solution
    results = scorer.score(model)

    print("\nResults:")
    print(f"Mean R² across all features: {results['mean_r2']:.6f}")
    print("\nR² for first 5 features:")
    for i in range(min(5, len(scorer.features))):
        feature = scorer.features[i]
        print(f"  {feature}: {results[feature]:.6f}")

    print(f"\nTotal features: {len(scorer.features)}")

    print("\n" + "=" * 60)
    print("Try submitting an archive with solution.py file")
    print("to test the solution submission mechanism!")
    print("=" * 60)


            
            

       