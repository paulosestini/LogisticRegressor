from LogisticRegressor import LogisticRegressor
import torch
import matplotlib.pyplot as plt

n_points_per_label = 50
label_0_points = 3*torch.rand(n_points_per_label) + 3
label_1_points = 3*torch.rand(n_points_per_label) + 5

label_0 = torch.zeros(n_points_per_label)
label_1 = torch.ones(n_points_per_label)

x = torch.cat((label_0_points, label_1_points), 0).view(-1, 1)
target = torch.cat((label_0, label_1), 0)

regressor = LogisticRegressor()
regressor.fit(x, target, n_iters=2000)

error = regressor.get_error()

x_test = torch.linspace(label_0_points.min(), label_1_points.max()).view(-1, 1)
predicted = regressor.predict(x_test)

plt.style.use('ggplot')
fig, ax = plt.subplots(2, 1)
ax[0].plot(x_test, predicted.detach(), color='black', label='Probability distribution')
ax[0].scatter(label_0_points, label_0, label="Class 0")
ax[0].scatter(label_1_points, label_1, label="Class 1")
ax[1].plot(error)
ax[0].legend()
ax[0].set_xlabel("Feature")
ax[0].set_ylabel("Class labels/Probability")
ax[1].set_xlabel("Iterations")
ax[1].set_ylabel("Loss (Binary Cross Entropy)")

plt.suptitle("Logistic Regression Example")
plt.show()