# perform Newton's method
W = np. zeros (X. shape [1])
W=W.reshape(3,1)

for i in range ( iterations ):
    y_pred = sigmoid ( np. dot (X , W))
    s = ( (y_pred - y) * (1 - y_pred - y))
    s = s.reshape(len(s))
    S = np.diag(s)
    mulfact_a = X.T @ S @ X
    mulfact_a = mulfact_a / y . size
    mulfact_a = np.linalg.inv(mulfact_a)
    loss = log_loss ( y , y_pred )
    loss_history_Newton . append ( np. sum ( np.mean(loss) ))
    residual = y_pred - y
    residual = residual.reshape(len(residual))
    diagonal_residual = np. diag ( residual )
    mulfact_b= one_matrix. T @ diagonal_residual@ X 
    mulfact_b = mulfact_b.T / y . size
    W -= mulfact_a @ mulfact_b
    print ( f'Iteration {i} : Loss {loss_history_Newton [i]}')