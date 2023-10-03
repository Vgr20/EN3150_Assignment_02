# Plot the loss function over iterations
plt. figure ()
plt. plot ( np. arange ( iterations ), loss_history_Newton )
plt. title ('Logistic Regression Loss per Iteration')
plt. xlabel ('Iterations')
plt. ylabel ('Loss')
plt. show ()