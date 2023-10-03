#performing Batch Gradient Descent
for i in range ( iterations ):
    y_pred = sigmoid ( np. dot (X , W))
    loss = log_loss ( y , y_pred )
    loss_history_BGD . append ( np. sum ( np.mean(loss) ))
    residual = y_pred - y
    residual = residual.reshape(len(residual))
    diagonal_residual = np. diag ( residual )
    gradient = one_matrix. T @ diagonal_residual@ X 
    gradient = gradient.T / y . size
    W -= learning_rate * gradient
    print ( f'Iteration {i} : Loss {loss_history_BGD [i]}')