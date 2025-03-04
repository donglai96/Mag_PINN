
"""
The input of the model: a, x, y
The output of the model : Bx, By

The pinn part(2d) : f = Bx_x + By_ y -> Nf points
The total mse composed of 3 parts:
1. regular loss part: (Bp - Bx)**2   p: predict
2. in-situ divergence pinn: ft = Bxp_x + Byp_ y t points: t means training points Nt points
3. fixed points of  pinn part(2d) : f = Bxf_x + Byf_ y -> Nf points

2 and 3 could put together
"""
import sys
import json
import os
import tensorflow as tf
import numpy as np
from neuralnetwork import NeuralNetwork


class BFieldInformedNN(NeuralNetwork):
    def __init__(self, hp, logger, X_f,ub,lb, alpha):
        super().__init__(hp,logger,ub,lb)

        self.a_f = self.tensor(X_f[:,0:1])
        self.x_f = self.tensor(X_f[:,1:2])
        self.y_f = self.tensor(X_f[:,2:3])
        self.alpha = alpha
    def bxy_model(self, X):
        a = X[:, 0:1]
        x = X[:, 1:2]
        y = X[:, 2:3]
        with tf.GradientTape(persistent= True) as tape:
            tape.watch(a)
            tape.watch(x)
            tape.watch(y)
            Xtemp = tf.concat([a,x,y],axis = 1)

            bxy = self.model(Xtemp)
            bx = bxy[:,0:1]
            by = bxy[:,1:2]

        bx_x = tape.gradient(bx, x)
        by_y = tape.gradient(by, y)
        del tape

        return bx, by, bx_x, by_y

    # Actual PINN
    def f_model(self):
        with tf.GradientTape(persistent= True) as tape:

            tape.watch(self.a_f)
            tape.watch(self.x_f)
            tape.watch(self.y_f)

            X_f = tf.concat([self.a_f,self.x_f,self.y_f], axis = 1)

            # Getting the prediction

            bx,by,bx_x,by_y = self.bxy_model(X_f)

        del tape

        div_b = bx_x + by_y
        return div_b

    def loss(self, bxy, bxy_pred):
        bx = bxy[:,0:1]
        by = bxy[:,1:2]
        bx_pred = bxy_pred[:, 0:1]
        by_pred = bxy_pred[:, 1:2]

        div_pred = self.f_model()
        mse_0 = tf.reduce_mean(tf.square(bx - bx_pred)) + \
                tf.reduce_mean(tf.square(by - by_pred))

        mse_f = tf.reduce_mean(tf.square(div_pred))

        tf.print(f"mse_0 {mse_0}    alpha {self.alpha}   mse_f    {mse_f}")

        return mse_0 + self.alpha * mse_f

    def predict(self, X_star):
        bxy_pred = self.model(X_star)
        bx_pred = bxy_pred[:,0:1]
        by_pred = bxy_pred[:,1:2]
        return bx_pred.numpy(), by_pred.numpy()




