# https://classroom.udacity.com/courses/ud953/lessons/4374471116/concepts/44678786200923
# 課程2 向量

from math import sqrt,acos,pi
from decimal import Decimal,getcontext
import numpy as np

getcontext().prec = 30
class Vector(object):
    CANNOT_NORMALIZE_ZERO_VECTOR_MSG = 'Cannot normalize the zero vector'
    NO_UNIQUE_PARALLEL_COMPONENT_MSG = 'NO_UNIQUE_PARALLEL_COMPONENT_MSG'
    NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG = 'NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG'
    ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG = 'ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG'

    def __init__(self,coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple([Decimal(x) for x in coordinates])
            #self.coordinates = [float(x) for x in coordinates]#改這樣時會回傳一個list，但line.py要改寫
            self.dimension = len(self.coordinates)

        except ValueError:
            raise ValueError('The coordinates must be nonempty')

        except TypeError:
            raise TypeError('The coordinates must be on iterable')
#輸出顯示計算結果的文字
    def __str__(self):
        return 'Vector: {}'.format(self.coordinates)

#輸出比較結果的bool值
    def __eq__(self,v):
        return self.coordinates == v.coordinates

#4 加減和標量乘法
    #加法
    def plus(self,v):
        new_coordinates = [x+y for x,y in zip(self.coordinates,v.coordinates)]
        # new_coordinates = []
        # n = len(self.coordinates)
        # for i in range(n):
        #     new_coordinates.append(self.coordinates[i] + v.coordinates[i])
        return Vector(new_coordinates)

    #減法
    def minus(self,v):
        new_coordinates = [x-y for x,y in zip(self.coordinates,v.coordinates)]
        return Vector(new_coordinates)

    #標量乘法
    def times_scalar(self,c):
        new_coordinates = [Decimal(c)*x for x in self.coordinates]
        return Vector(new_coordinates)

#6 大小和方向函數
    #大小
    def magnitude(self):
        coordinates_squared = [x**2 for x in self.coordinates]
        #return sqrt(sum(coordinates_squared))
        return Decimal(sum(coordinates_squared)).sqrt() #與教學不同處

    #方向
    def normalized(self):
        try:
            magnitude = self.magnitude()
            return self.times_scalar(Decimal('1.0')/Decimal(magnitude))
        except ZeroDivisionError:
            raise Exception(self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG)

#8 點積和夾角函數
    #點積
    def dot(self,v):
        return sum([x*y for x,y in zip(self.coordinates,v.coordinates)])

    #內夾角
    def angle_with(self,v,in_degrees=False):
        try:
            u1 = self.normalized()
            u2 = v.normalized()
            angle_in_radians = acos(u1.dot(u2))

            if in_degrees:
                degrees_per_radian = 180 / pi
                return angle_in_radians * degrees_per_radian
            else:
                return angle_in_radians

        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception('Cannot compute an angle with the zero vector')
            else:
                raise e

#10 檢查是否平行或正交
    #為正交
    def is_orthogonal_to(self,v,tolerance=1e-10):
        return abs(self.dot(v)) < tolerance

    #為平行
    def is_parallel_to(self,v):
        return (self.is_zero() or v.is_zero() or self.angle_with(v) == 0 or self.angle_with(v) == pi)

    #為零
    def is_zero(self,tolerance=1e-10):
        return self.magnitude() < tolerance

#12 向量投影函數
    #正交
    def component_orthogonal_to(self,basis):
        try:
            projection = self.component_parallel_to(basis)
            return self.minus(projection)
        except Exception as e:
            if str(e) == self.NO_UNIQUE_PARALLEL_COMPONENT_MSG:
                raise Exception(self.NO_UNIQUE_ORTHOGONAL_COMPONENT_MSG)
            else:
                raise e

    #平行
    def component_parallel_to(self,basis):
        try:
            u = basis.normalized()
            weight = self.dot(u)
            return u.times_scalar(weight)
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR_MSG:
                raise Exception(self.NO_UNIQUE_PARALLEL_COMPONENT_MSG)
            else:
                raise e

#14 向量積函數
    #三角形面積
    def area_of_triangle_with(self,v):
        return self.area_of_parallelogram_with(v) / Decimal('2.0')

    #平行四邊形面積
    def area_of_parallelogram_with(self,v):
        cross_product = self.cross(v)
        return cross_product.magnitude()

    #向量積
    def cross(self,v):
        try:
            x_1,y_1,z_1 = self.coordinates
            x_2,y_2,z_2 = v.coordinates
            new_coordinates = [ y_1 * z_2 - y_2 * z_1 ,
                              -(x_1 * z_2 - x_2 * z_1),
                                x_1 * y_2 - x_2 * y_1  ]
            return Vector(new_coordinates)
        except ValueError as e:
            msg = str(e)
            if msg == 'nees more than 2 values to unpack':
                self_embedded_in_R3 = Vector(self.coordinates + ('0',))
                v_embedded_inR3 = Vector(v.coordinates + ('0',))
                return self_embedded_in_R3
            elif (msg == 'too many values to unpack' or msg == 'need more than 1 value to unpack'):
                raise Exception(self.ONLY_DEFINED_IN_TWO_THREE_DIMS_MSG)
            else:
                raise e

#2.Vector模塊
# my_vector = Vector([1,2,3])
# print(my_vector) #Vector: (1, 2, 3)

# my_vector2 = Vector([-1,2,3])
# print(my_vector == my_vector2) #False



# 4.練習加減和標量乘法
# v = Vector([8.218,-9.341])
# w = Vector([-1.129,2.111])
# print(v.plus(w))

# v = Vector([7.119,8.215])
# w = Vector([-8.223,0.878])
# print(v.minus(w))

# v = Vector([1.671,-1.012,-0.318])
# c = 7.41
# print(v.times_scalar(c))



#6.練習編寫大小和方向函數
# v = Vector([-0.221,7.437])
# print(v.magnitude())

# v = Vector([8.813,-1.331,-6.247])
# print(v.magnitude())

# v = Vector([5.581,-2.136])
# print(v.normalized())

# v = Vector([1.996,3.108,-4.554])
# print(v.normalized())


#8.練習編寫點積和夾角函數
# v = Vector(['7.887','4.138'])
# w = Vector(['-8.802','6.776'])
# print(v.dot(w))

# v = Vector(['-5.955','-4.904','-1.874'])
# w = Vector(['-4.496','-8.755','7.103'])
# print(v.dot(w))

# v = Vector(['3.183','-7.627'])
# w = Vector(['-2.668','5.319'])
# print(v.angle_with(w))

# v = Vector(['7.35','0.221','5.188'])
# w = Vector(['2.751','8.259','3.985'])
# print(v.angle_with(w,in_degrees=True))



#10.練習檢查是否平行或正交
# print('first pair...')
# v = Vector(['-7.579','-7.88'])
# w = Vector(['22.737','23.64'])
# print('is parallel:',v.is_parallel_to(w))
# print('is orthogonal:',v.is_orthogonal_to(w))

# print('second pair...')
# v = Vector(['-2.029','9.97','4.172'])
# w = Vector(['-9.231','-6.639','-7.245'])
# print('is parallel:',v.is_parallel_to(w))
# print('is orthogonal:',v.is_orthogonal_to(w))

# print('third pair...')
# v = Vector(['-2.328','-7.284','-1.214'])
# w = Vector(['-1.821','1.072','-2.94'])
# print('is parallel:',v.is_parallel_to(w))
# print('is orthogonal:',v.is_orthogonal_to(w))

# print('fourth pair...')
# v = Vector(['2.118','4.827'])
# w = Vector(['0','0'])
# print('is parallel:',v.is_parallel_to(w))
# print('is orthogonal:',v.is_orthogonal_to(w))



#12練習編寫向量投影函數
# print('#1')
# v = Vector(['3.039','1.879'])
# w = Vector(['0.825','2.036'])
# print(v.component_parallel_to(w))

# print('\n#2')
# v = Vector(['-9.88','-3.264','-8.159'])
# w = Vector(['-2.155','-9.353','-9.473'])
# print(v.component_orthogonal_to(w))

# print('\n#3')
# v = Vector(['3.009','-6.172','3.692','-2.51'])
# w = Vector(['6.404','-9.144','2.759','8.718'])
# vpar = v.component_parallel_to(w)
# vort = v.component_orthogonal_to(w)
# print('parallel component:',vpar)
# print('orthogonal component:',vort)



#14練習編寫向量積函數
v = Vector(['8.462','7.893','-8.187'])
w = Vector(['6.984','-5.975','4.778'])
print('#1:',v.cross(w))

v = Vector(['8.987','9.838','5.031'])
w = Vector(['-4.268','-1.861','-8.866'])
print('#2:',v.area_of_parallelogram_with(w))

v = Vector(['1.5','9.547','3.691'])
w = Vector(['-6.007','0.124','5.772'])
print('#3:',v.area_of_triangle_with(w))

