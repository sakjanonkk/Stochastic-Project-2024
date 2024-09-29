import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import binom

#******* intitial parameter *****

x1 = 94
x2 = 96

N = 10 
p = 0.31 + (x2/1000) 
n = 10000  

#************* ข้อ2 ************

X = np.random.binomial(N, p, n)
minX = X.min()
maxX = X.max()

e = np.arange(minX - 0.5, maxX + 1)     # edge
nbins = len(e) - 1                      # bins
Mn = np.mean(X)
pn = Mn / N
np.histogram(X, bins=e)

# Plot Histogram จากข้อมูล X
plt.hist(X, bins=e, align='left',fill=False)
plt.title(f"Histogram with bins={nbins} \n Mean: {Mn} \n Probability: {pn}")
plt.axvline(Mn, label=f"Mean = {Mn}", color='orange')
plt.legend()
print(f"\n\n\nbins: {np.histogram(X, bins=e)[0]}  \nEdge: {np.histogram(X, bins=e)[1]}") 

#******* SourceCode 1.3 *********
k = np.arange(0, N+1) # กำหนดช่วงของค่า k สำหรับการคำนวณ PMF
PMF = binom.pmf(k, N, pn) # คำนวณ PMF สำหรับ Binomial Distribution

# สร้างกราฟ PMF
fig, ax = plt.subplots(1, 1)
ax.plot(k, PMF, 'ro', ms=12, mec='r')  # จุด
ax.vlines(k, 0, PMF, colors='r', lw=4)  # เส้นแนวตั้ง
plt.title("PMF of Binomial Distribution")
plt.grid()

#************* ข้อ 3 ************
# _______ SouceCode ของอาจารย์ ข้อ 3 ______
# _______ Candidate PMF ________________
H = list(np.bincount(X))  # เอา x จากข้อ 2 
for j in range(len(H), N + 1):
    H.append(0) 
Z = 0
for j in range(N + 1):
    Z += (H[j] - n * PMF[j]) ** 2 / (n * PMF[j]) # คำนวณ Z-statistic ตามสูตร lecture 16
#_____ หา degree of freedom และ Z𝜶 _____
#--- degree of freedom ---
m = nbins                                       # number of bins
r = 2                                           # จำนวน parameter ของ binomial (n,p)
dof_k = m - 1 - r                               # degree of freedom (k)
#--- Z𝜶 ---
alpha = 0.05                                    # significance level = 5% , 𝜶 = 5%
Z_alpha = stats.chi2.ppf(1-alpha, dof_k)          # Z𝜶
#_____ ตรวจสอบความ goodness ของ candidate PMF ______
check_goodness = ""
compareZ = ""

if Z < Z_alpha:
    goodness = "\033[4mgood fit to the data\033[0m"
    compareZ = "(Z < Z𝜶)"
else:
    goodness = "\033[4mnot good fit to the data\033[0m"
    compareZ = "(Z > Z𝜶)"
    
    
#************* ข้อ 4 ************
Sn = np.std(X,ddof=1)        #           
dofํ_Y_alpha_over_2 = n - 1   #degree of freedom (v)
y_alpha_over_2 = stats.t.ppf(1 - alpha/2, dofํ_Y_alpha_over_2)

# คำนวณ confidence interval
lower_bound = Mn - y_alpha_over_2 * (Sn / np.sqrt(n))
upper_bound = Mn + y_alpha_over_2 * (Sn / np.sqrt(n))


# Summary
print("************ Histogram ******************\n*")
print(f"* min(X)                                   : {minX} ")
print(f"* max(X)                                   : {maxX} ")
print(f"* bins                                     : {nbins} bins")
print(f"* Mean                                     : {Mn:.5f}")
print(f"* Probability                              : {pn:.5f} ({(pn * 100):.1f}%)\n*")


print("***** Candidate PMF is goodness or not ****\n*")
print(f"* Degree of freedom (k)                    : {dof_k}")
print(f"* a statistics (Z)                         : {Z:.5f}")
print(f"* threshold (Z𝜶)                           : {Z_alpha:.5f}")
print(f"* {compareZ} so candidate pmf is {goodness}.\n*")


print("**** Confidence interval for the mean ******\n*")
print(f"* Sn (σ)                                   : {Sn:.5f}")
print(f"* Degree of freedom (v = n - 1)            : {dofํ_Y_alpha_over_2}")
print(f"* Y𝜶/2                                     : {y_alpha_over_2:.5f}")
print(f"* Confidence Interval for the mean         : [{lower_bound:.5f}, {upper_bound:.5f}]") 

# plt.show() 