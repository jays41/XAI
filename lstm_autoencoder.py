def sigmoid(x): pass
def tanh(x): pass

def lstm_cell_forward(x_t, h_prev, c_prev, params): pass
def init_lstm_params(input_dim, hidden_dim): pass

def lstm_encoder_forward(X, params): pass
def lstm_decoder_forward(X, h_init, c_init, params): pass

def lstm_autoencoder_forward(X, encoder_params, decoder_params): pass

def compute_loss(y_true, y_pred): pass
def backward_pass(...): pass
def update_params(params, grads, lr): pass
