import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
#import seaborn as sns
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
    transition = pd.read_excel(path, sheet_name = 3, skiprows=1, index_col = 0)
    transition = transition*100
    return transition

def readforward (path):
    forward = pd.read_excel(path, sheet_name = 4, skiprows=1, index_col = 0)
    forward = forward*100
    forward.columns = range(0, 6)
    return forward

def readRR(path):
    RR = pd.read_excel(path, sheet_name = 5, skiprows=1, index_col = 0)
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

def readSimulations(path):
    Level = pd.read_excel(path, sheet_name = 2, skiprows=6, usecols="L", nrows=1, header=None, names=["Value"]).iloc[0]["Value"]
    if Level == "Ultra High":
        return 1500000, Level
    elif Level == "Very High":
        return 600000, Level
    elif Level == "High":
        return 300000, Level
    elif Level == "Normal":
        return 150000, Level
    elif Level == "Low":
        return 70000, Level
    else:
        return 20000, Level


#----------------------------------------------------------------------------------------------------------------

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'BondPortfolio.xlsm')

#number of simulations
n, SimLevel = readSimulations(file_path)


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



avg = np.mean(portfolio)
plt.figure()
#sns.kdeplot(portfolio, bw_adjust=0.9, label="Kernel Density Estimate")
plt.hist(portfolio, bins=300, density=True, color="skyblue")
plt.axvline(avg, color='red', linestyle='--', label="Mean Value: " + "{:,.3f}".format(np.mean(portfolio)))
plt.title("1Y Portfolio Value Distribution")
plt.xlabel("Portfolio Value")
plt.ylabel("Density")

plt.xlim(np.percentile(portfolio, 0.1),)
plt.legend()

#--------------- Specify Var Level
VarLvl = 95
VarLvl2 = 99
#---------------

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
print("Number of Simulations:",SimLevel)
print("\n---- Portfolio Statistics ----\n")
print("Mean =", f"{round(np.mean(portfolio),3):,}")
print("S.D =", f"{round(np.std(portfolio),3):,}")
print("1% percentile =", f"{round(np.percentile(portfolio, 1), 3):,}")
print("5% percentile =", f"{round(np.percentile(portfolio, 5), 3):,}")
print("95% percentile =", f"{round(np.percentile(portfolio, 95), 3):,}")
print("99% percentile =", f"{round(np.percentile(portfolio, 99), 3):,}")

print("\n---- Loss Statistics ----\n")
print(str(VarLvl)+"% 1Y Rel. VaR = ", f"{round(var,3):,}")
print("Expected Shortfall = ", f"{round(np.mean(ES),3):,}")
print()
print(str(VarLvl2)+"% 1Y Rel. VaR = ", f"{round(var2,3):,}")
print("Expected Shortfall = ", f"{round(np.mean(ES2),3):,}")


plt.show()
