#!/usr/bin/env python3
# by frank tian on 7.17.2020

import math
class quaternion:
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z
        self.vector = [x, y, z]
        self.all = [w, x, y, z]

    def __str__(self):
        op = [" ", "i ", "j ", "k"]
        q = self.all.copy()
        result = ""
        for i in range(4):
            if q[i] < -1e-8 or q[i] > 1e-8:
                result = result + str(round(q[i], 4)) + op[i]
        if result == "":
            return "0"
        else:
            return result

    def __add__(self, quater):
        q = self.all.copy()
        for i in range(4):
            q[i] += quater.all[i]
        return quaternion(q[0], q[1], q[2], q[3])

    def __sub__(self, quater):
        q = self.all.copy()
        for i in range(4):
            q[i] -= quater.all[i]
        return quaternion(q[0], q[1], q[2], q[3])

    def __mul__(self, quater):
        q = self.all.copy()
        p = quater.all.copy()
        s = q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3]
        x = q[0] * p[1] + q[1] * p[0] + q[2] * p[3] - q[3] * p[2]
        y = q[0] * p[2] - q[1] * p[3] + q[2] * p[0] + q[3] * p[1]
        z = q[0] * p[3] + q[1] * p[2] - q[2] * p[1] + q[3] * p[0]
        return quaternion(s, x, y, z)

    def divide(self, quaternion):
        result = self * quaternion.inverse()
        return result

    def modpow(self):
        q = self.all
        return sum([i ** 2 for i in q])

    def mod(self):
        return pow(self.modpow(), 1 / 2)

    def conj(self):
        q = self.all.copy()
        for i in range(1, 4):
            q[i] = -q[i]
        return quaternion(q[0], q[1], q[2], q[3])

    def inverse(self):
        q = self.all.copy()
        mod = self.modpow()
        for i in range(4):
            q[i] /= mod
        return quaternion(q[0], -q[1], -q[2], -q[3])

class bullet_quaternion(quaternion):
    def __init__(self, *args):
        # [x,y,z,w] -> [w,x,y,z]
        if len(args) == 1:
            super(bullet_quaternion, self).__init__(args[0][3], args[0][0], args[0][1], args[0][2])
        # theta, u
        if len(args) == 2:
            theta = args[0]
            u = args[1]
            u_mod = math.sqrt(u[0]**2 + u[1]**2 + u[2]**2) if math.sqrt(u[0]**2 + u[1]**2 + u[2]**2) > 1e-10 else 1
            u = [u[0]/u_mod, u[1]/u_mod, u[2]/u_mod]
            super(bullet_quaternion, self).__init__(math.cos(theta/2), math.sin(theta/2)*u[0], math.sin(theta/2)*u[1], math.sin(theta/2)*u[2])
        # x,y,z,w -> [w,x,y,z]
        elif len(args) == 4:
            super(bullet_quaternion, self).__init__(args[3], args[0], args[1], args[2])
    def return_bullet_quaternion(self):
        return [self.all[1],self.all[2],self.all[3],self.all[0]]

    def ori(self, ini_ori):
        quaternion_ori = bullet_quaternion(ini_ori[0],ini_ori[1],ini_ori[2],0)
        rotated_ori = self * quaternion_ori  * self.inverse()
        return [rotated_ori.all[1],rotated_ori.all[2],rotated_ori.all[3]]

if __name__ == "__main__":
    pass