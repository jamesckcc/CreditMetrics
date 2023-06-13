import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def estimate_alpha_beta(mean, std_dev):
    variance = std_dev ** 2
    factor = mean * (1 - mean) / variance - 1
    alpha = mean * factor
    beta = (1 - mean) * factor
    result = [alpha, beta]
    return result

def create_alpha_beta_dict(seniority):
    alpha_beta_dict = {key: estimate_alpha_beta(value[0] / 100, value[1] / 100) for key, value in seniority.items()}
    return alpha_beta_dict

def transprob(df, initial, final):
    return df.at[initial, final]

def bond1y(df, rating, M, C, FV):
    Coupon = C/100
    value = FV*Coupon
    for i in range(M-1):
        value += (FV*Coupon)/(1+df.at[rating,i]/100)**(i+1)
    
    value += (FV)/(1+df.at[rating,i]/100)**(i+1)
    return value
    
    
class Bond:
    def __init__(self, Rating, SeniorityClass, M, Coupon, FV):
        self.Rating = Rating
        self.SeniorityClass = SeniorityClass
        self.M = M
        self.Coupon = Coupon
        self.FV = FV
        
    def bondinfo(self, df1, df2, df3):
        RatingInfo = []
        Ratings = ["AAA","AA","A","BBB","BB", "B","CCC"]
        cum = 100
        for i in Ratings:
            bondval = bond1y(df2, i, self.M, self.Coupon, self.FV)
            Prob = transprob(df1, self.Rating, i)
            Threshold = stats.norm.ppf(cum/100)
            cum -= Prob
            RatingInfo.append([i,bondval,Threshold, Prob])
            
        DThreshold = stats.norm.ppf(cum/100)    
        RatingInfo.append(["Default", 0, DThreshold])
            
        RR = df3[self.SeniorityClass]
        
        Info = [RatingInfo, RR, self.FV]
            
        return Info
    

def simulatedprice(z, bondinfo):
    price = []
    for i in z:
        for j in range(len(bondinfo[0])):
            if i >= bondinfo[0][j][2]:
                price.append(bondinfo[0][j-1][1])
                break
        else:
            price.append(bondinfo[2] * np.random.beta(bondinfo[1][0], bondinfo[1][1]))
    return price

#Assume There are only ratings AAA, AA, A, BBB, BB, B, CCC, Default
OneYearTransitiondict = {
    "AAA":[90.81,8.33,0.68,0.06,0.12,0,0,0], 
    "AA":[0.70,90.65,7.79,0.64,0.06,0.14,0.02,0], 
    "A":[0.09,2.27,91.05,5.52,0.74,0.26,0.01,0.06], 
    "BBB":[0.02,0.33,5.95,86.93,5.30,1.17,0.12,0.18], 
    "BB":[0.03,0.14,0.67,7.73,80.53,8.84,1,1.06], 
    "B":[0,0.11,0.24,0.43,6.48,83.46,4.07,5.2], 
    "CCC":[0.22,0,0.22,1.3,2.38,11.24,64.86,19.79]
    }

OneYearTransition = pd.DataFrame(OneYearTransitiondict, index = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "Default"])
OneYearTransition = OneYearTransition.T



OneYearForwardZeroCurvesdict = {
    "AAA":[3.6,4.17,4.73,5.12,5.51,5.9], 
    "AA":[3.65,4.22,4.78,5.17,5.56,5.95], 
    "A":[3.72,4.32,4.93,5.32,5.71,6.10], 
    "BBB":[4.1,4.67,5.25,5.63,6.01,6.39], 
    "BB":[5.55,6.02,6.78,7.27,7.76,8.25], 
    "B":[6.05,7.02,8.03,8.52,9.01,9.5], 
    "CCC":[15.05,15.02,14.03,13.52,13.01,12.50]
    }

OneYearForwardZeroCurves = pd.DataFrame(OneYearForwardZeroCurvesdict)
OneYearForwardZeroCurves = OneYearForwardZeroCurves.T


Seniority = {
    "Senior Secured": [53.8,26.86], 
    "Senior Unsecured": [51.13, 25.45], 
    "Senior Subordinated": [38.52,23.81], 
    "Subordinated": [32.74,20.18], 
    "Junior Subordinated": [17.09, 10.9]
    }

SeniorityPara = create_alpha_beta_dict(Seniority)


df1 = OneYearTransition
df2 = OneYearForwardZeroCurves
df3 = SeniorityPara


#Bonds
NumberOfBonds = 10
BondA = Bond("A", "Senior Secured", 5,6,3000000)
BondB = Bond("BB", "Senior Unsecured", 4,7,1000000)
BondC = Bond("AA", "Senior Secured", 3,5,4200000)
BondD = Bond("B", "Senior Unsecured", 6,2,1000000)
BondE = Bond("CCC", "Subordinated", 2,8,1100000)
BondF = Bond("BBB", "Senior Unsecured", 4,3,3000000)
BondG = Bond("AAA", "Senior Secured", 3,4,2000000)
BondH = Bond("A", "Senior Secured", 3,5,1900000)
BondI = Bond("CCC", "Junior Subordinated", 2,4,1000000)
BondJ = Bond("BBB", "Senior Secured", 3,5,2000000)



#Correlation
correlation = [
[1.00,  0.82,  0.45,  0.71,  0.89,  0.65,  0.23,  0.79,  0.76,  0.68],
[0.82,  1.00,  0.51,  0.78,  0.81,  0.61,  0.29,  0.75,  0.73,  0.70],
[0.45,  0.51,  1.00,  0.53,  0.47,  0.56,  0.05,  0.49,  0.52,  0.59],
[0.71,  0.78,  0.53,  1.00,  0.69,  0.72,  0.14,  0.67,  0.64,  0.66],
[0.89,  0.81,  0.47,  0.69,  1.00,  0.63,  0.21,  0.77,  0.74,  0.60],
[0.65,  0.61,  0.56,  0.72,  0.63,  1.00,  0.10,  0.62,  0.59,  0.57],
[0.23,  0.29,  0.05,  0.14,  0.21,  0.10,  1.00,  0.25,  0.22,  0.19],
[0.79,  0.75,  0.49,  0.67,  0.77,  0.62,  0.25,  1.00,  0.72,  0.64],
[0.76,  0.73,  0.52,  0.64,  0.74,  0.59,  0.22,  0.72,  1.00,  0.61],
[0.68,  0.70,  0.59,  0.66,  0.60,  0.57,  0.19,  0.64,  0.61,  1.00]
]

n = 100000

mean = [0]*NumberOfBonds
samples = np.random.multivariate_normal(mean, correlation, size = n)
samplesdf = pd.DataFrame(samples)

#Initiation
priceA = simulatedprice(samplesdf[0].tolist(), BondA.bondinfo(df1, df2, df3))
priceB = simulatedprice(samplesdf[1].tolist(), BondB.bondinfo(df1, df2, df3))
priceC = simulatedprice(samplesdf[2].tolist(), BondC.bondinfo(df1, df2, df3))
priceD = simulatedprice(samplesdf[3].tolist(), BondD.bondinfo(df1, df2, df3))
priceE = simulatedprice(samplesdf[4].tolist(), BondE.bondinfo(df1, df2, df3))
priceF = simulatedprice(samplesdf[5].tolist(), BondF.bondinfo(df1, df2, df3))
priceG = simulatedprice(samplesdf[6].tolist(), BondG.bondinfo(df1, df2, df3))
priceH = simulatedprice(samplesdf[7].tolist(), BondH.bondinfo(df1, df2, df3))
priceI = simulatedprice(samplesdf[8].tolist(), BondI.bondinfo(df1, df2, df3))
priceJ = simulatedprice(samplesdf[9].tolist(), BondJ.bondinfo(df1, df2, df3))


portfolio = []

for i in range(n):
    portfolio.append(priceA[i] + priceB[i] + priceC[i]+ priceD[i]+ priceE[i]+ priceF[i]+ priceG[i]+ priceH[i]+ priceI[i]+ priceJ[i])
    

portfolioloss = np.mean(portfolio) - np.array(portfolio) 


plt.figure()
sns.kdeplot(portfolio, bw_adjust=0.17)
plt.title("Portoflio Value")
plt.xlabel("Portfolio Value")
plt.xlim(16000000,)
plt.show()

plt.figure()
sns.kdeplot(portfolioloss, bw_adjust=0.17)
plt.title("Relative Loss")
plt.xlabel("Portfolio Loss")
plt.xlim(None,3000000)
plt.show()

ES = []

#--------------- Specify Var Level
VarLvl = 95
#---------------


avg = np.mean(portfolio)
var = np.percentile(portfolioloss, VarLvl)



ES = []
for i in portfolio:
    if avg - i > var:
        ES.append(avg - i)
        

#-------------------------------------------------------------------------------



print("----Portflio Statistics----\n")
print("Mean =",np.mean(portfolio))
print("S.D =", np.std(portfolio))
print("1% percentile =", np.percentile(portfolio, 1))
print("5% percentile =", np.percentile(portfolio, 5))
print("95% percentile =", np.percentile(portfolio, 95))
print("99% percentile =", np.percentile(portfolio, 99))

print("\n----Loss Statistics----\n")
print(str(VarLvl)+"% Rel. VaR = ", var)
print("Expected Shortfall = ", np.mean(ES))








