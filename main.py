import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
#from collections import Counter


def transprob(df, initial, final):
    return df.at[initial, final]

def bond1y(df, rating, M, C, FV):
    Coupon = C/100
    value = FV*Coupon
    k = 0  
    for k in range(M-1):
        value += (FV*Coupon)/(1+df.at[rating, k]/100)**(k+1)
    
    value += (FV)/(1+df.at[rating, k]/100)**(k+1)
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
        Ratings = ["AAA","AA","A","BBB","BB","B","CCC"]
        cum = 100
        for c in Ratings:
            bondval = bond1y(df2, c, self.M, self.Coupon, self.FV)
            Prob = transprob(df1, self.Rating, c)
            Threshold = stats.norm.ppf(cum/100)
            cum -= Prob
            RatingInfo.append([c,bondval,Threshold, Prob])
            
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


def readtransition (path):
    transition = pd.read_excel(file_path, sheet_name = 2, skiprows=1, index_col = 0)
    transition = transition*100
    return transition

def readforward (path):
    forward = pd.read_excel(file_path, sheet_name = 3, skiprows=1, index_col = 0)
    forward = forward*100
    forward.columns = range(0, 6)
    return forward

def readRR(path):
    RR = pd.read_excel(file_path, sheet_name = 4, skiprows=1, index_col = 0)
    seniority = RR.to_dict(orient='index')
    
    alpha_beta_dict = {
        key: [
            value['Mean'] * (value['Mean'] * (1 - value['Mean']) / (value['SD'] ** 2) - 1) / 100,
            (1 - value['Mean']) * (value['Mean'] * (1 - value['Mean']) / (value['SD'] ** 2) - 1) / 100
        ]
        for key, value in seniority.items()
    }
    alpha_beta_dict = {key: [value[0] * 100, value[1] * 100] for key, value in alpha_beta_dict.items()}
    return alpha_beta_dict

def readCorr(path):
    corr = pd.read_excel(path, sheet_name = 1, skiprows=2, index_col = 0)
    corr = corr.round(2)
    return corr.values.tolist()

def readPortfolio(path):
    port = pd.read_excel(path, usecols="A:E", skiprows=1)
    port = port.dropna()
    n = len(port.index)
    port = port.values.tolist()
    return port, n

#----------------------------------------------------------------------------------------------------------------

file_path = "BondPortfolio.xlsm"

#number of simulations
n = 200000


df, NumberOfBonds = readPortfolio(file_path)
df1 = readtransition(file_path)
df2 = readforward(file_path)
df3 = readRR(file_path)
correlation = readCorr(file_path)

Bonds = []

for i in df:
    Bonds.append(Bond(i[1], i[2], int(i[4]), round(i[3]*100), i[0]))

mean = [0]*NumberOfBonds
samples = np.random.multivariate_normal(mean, correlation, size = n)
samplesdf = pd.DataFrame(samples)

Prices = []

count = 0
for i in Bonds:
    Prices.append(simulatedprice(samplesdf[count].tolist(), i.bondinfo(df1, df2, df3)))
    count += 1

portfolio = []

for i in range(n):
    sum = 0
    for j in Prices:
        sum += j[i]
    portfolio.append(sum)
    

portfolioloss = np.mean(portfolio) - np.array(portfolio) 





#plt.figure()
#sns.kdeplot(portfolioloss, bw_adjust=0.26)
#plt.title("Relative Loss")
#plt.xlabel("Portfolio Loss")

plt.show()

ES = []

#--------------- Specify Var Level
VarLvl = 95
VarLvl2 = 99
#---------------


avg = np.mean(portfolio)
plt.figure()
sns.kdeplot(portfolio, bw_adjust=0.4)
plt.axvline(avg, color='red', linestyle='--', label="Mean Value: " + "{:,.3f}".format(np.mean(portfolio)))
plt.title("Portoflio Value Distribution")
plt.xlabel("Portfolio Value")

plt.xlim(np.percentile(portfolio, 0.1),)
plt.legend()

var = np.percentile(portfolioloss, VarLvl)
var2 = np.percentile(portfolioloss, VarLvl2)


ES = []
for i in portfolio:
    if avg - i > var:
        ES.append(avg - i)
        
ES2 = []
for i in portfolio:
    if avg - i > var2:
        ES2.append(avg - i)

#-------------------------------------------------------------------------------


print("\n***************************************************")
print("***************************************************")
print("\n----Portflio Statistics----\n")
print("Mean =", f"{round(np.mean(portfolio),3):,}")
print("S.D =", f"{round(np.std(portfolio),3):,}")
print("1% percentile =", f"{round(np.percentile(portfolio, 1), 3):,}")
print("5% percentile =", f"{round(np.percentile(portfolio, 5), 3):,}")
print("95% percentile =", f"{round(np.percentile(portfolio, 95), 3):,}")
print("99% percentile =", f"{round(np.percentile(portfolio, 99), 3):,}")

print("\n----Loss Statistics----\n")
print(str(VarLvl)+"% 1Y Rel. VaR = ", f"{round(var,3):,}")
print(str(VarLvl)+"% 1Y VaR Expected Shortfall = ", f"{round(np.mean(ES),3):,}")
print()
print(str(VarLvl2)+"% 1Y Rel. VaR = ", f"{round(var2,3):,}")
print(str(VarLvl2)+"% 1Y VaR Expected Shortfall = ", f"{round(np.mean(ES2),3):,}")


plt.show()





