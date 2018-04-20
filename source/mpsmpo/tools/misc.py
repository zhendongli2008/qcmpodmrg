import math

def binomial(k,n):
   binom = math.factorial(k)/(math.factorial(n)*math.factorial(k-n))
   return binom


if __name__ == '__main__':
   print binomial(10,3)    # 120
   print binomial(10,0)    # 1
   print binomial(20,10)   # 184756
   print binomial(40,20)   # 137846528820
   print binomial(100,50)  # 100891344545564193334812497256
