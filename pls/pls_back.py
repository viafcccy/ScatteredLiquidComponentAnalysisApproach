
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
 
from scipy.signal import savgol_filter
from sys import stdout

def base_pls_cv(X,y,n_components, return_model=False):
 
    # Simple PLS
    pls_simple = PLSRegression(n_components=n_components)
    # Fit
    pls_simple.fit(X, y)
    # Cross-validation
    y_cv = cross_val_predict(pls_simple, X, y, cv=10)
 
    # Calculate scores
    score = r2_score(y, y_cv)
    rmsecv = np.sqrt(mean_squared_error(y, y_cv))
 
    if return_model == False:
        return(y_cv, score, rmsecv)
    else:
        return(y_cv, score, rmsecv, pls_simple)
 
def pls_optimise_components(X, y, npc):
 
    rmsecv = np.zeros(npc)
    for i in range(1,npc+1,1):
 
        # Simple PLS
        pls_simple = PLSRegression(n_components=i)
        # Fit
        pls_simple.fit(X, y)
        # Cross-validation
        y_cv = cross_val_predict(pls_simple, X, y, cv=10)
 
        # Calculate scores
        score = r2_score(y, y_cv)
        rmsecv[i-1] = np.sqrt(mean_squared_error(y, y_cv))
 
    # Find the minimum of ther RMSE and its location
    opt_comp, rmsecv_min = np.argmin(rmsecv),  rmsecv[np.argmin(rmsecv)]
 
    return (opt_comp+1, rmsecv_min)
 
def regression_plot(y_ref, y_pred, title = None, variable = None):
 
    # Regression plot
    
    z = np.polyfit(y_ref, y_pred, 1)
    with plt.style.context(('seaborn')):
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(y_ref, y_pred, c='red', edgecolors='k')
        ax.plot(y_ref, z[1]+z[0]*y_ref, c='blue', linewidth=1)
        ax.plot(y_ref, y_ref, color='green', linewidth=1)
 
        if title is not None:
            plt.title(title, fontsize=20)
        if variable is not None:
            plt.xlabel('Measured ' + variable, fontsize=20)
            plt.ylabel('Predicted ' + variable, fontsize=20)
 
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()

raw = pd.read_excel("File_S1.xlsx")
 
X = raw.values[1:,14:].astype('float32')
y = raw.values[1:,1].astype('float32')
 
wl = np.linspace(350,2500, num=X.shape[1], endpoint=True)
X = savgol_filter(X, 11, polyorder = 2,deriv=0)
 
plt.figure(figsize=(12,8))
with plt.style.context(('seaborn')):
    plt.plot(wl, X.T, linewidth=1)
    plt.xlabel('Wavelength (nm)', fontsize=20)
    plt.ylabel('Reflectance', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
plt.show()

# Resample spectra into wavelength bands. 
Xr = X.reshape(X.shape[0],42,5).sum(axis=2)
wlr = wl.reshape(42,5).mean(axis=1)

# Find the optimal number of components, up to 20
opt_comp, rmsecv_min = pls_optimise_components(X, y, 20)
# Run a simple PLS model with the optimal number of components
predicted, r2cv_base, rmscv_base = base_pls_cv(X, y, opt_comp)
# Print metrics
print("PLS results:")
print("   R2: %5.3f, RMSE: %5.3f"%(r2cv_base, rmscv_base))
print("   Number of Latent Variables: "+str(opt_comp))
# Plot result
regression_plot(y, predicted, title="Basic PLS", variable = "TOC (g/kg)")

# Define an empty list, to be populated with the RMSE values obtained by eliminating each band in sequance
rmscv = []
# Define a Leave-One-Out cross-validator
loo = LeaveOneOut()
# Loop over the bands one by one
for train_wl, test_wl in loo.split(wlr):
    
    # Optimise and fit a PLS model with one band removed
    opt_comp, rmsecv_min = pls_optimise_components(Xr[:,train_wl], y, 10)
    predicted, r2cv_loo, rmscv_loo = base_pls_cv(Xr[:,train_wl], y, opt_comp)
    # Append the value of the RMSE to the list
    rmscv.append(rmscv_loo)
    
    # Print a bunch of stuff
    stdout.write("\r"+str(test_wl))
    stdout.write(" ")
    stdout.write(" %1.3f" %rmscv_loo)
    stdout.write(" ")
    stdout.write(" %1.3f" %r2cv_loo)
    stdout.write(" ")
    stdout.flush()
 
stdout.write('\n')
# List to array
rmscv = np.array(rmscv)

plt.figure(figsize=(12,8))
with plt.style.context(('seaborn')):
    # Plot the RMSE obtained by eliminating each band
    plt.plot(rmscv, 'o--', linewidth=2)
    # Plot the baseline RMSE-CV from the previous step
    plt.plot([0, 42], [rmscv_base, rmscv_base], 'r', lw=2)
    
    plt.xlabel('Index of Band', fontsize=20)
    plt.ylabel('RMSE-CV', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
plt.show()

# Resample spectra
Xr_optim = X.reshape(X.shape[0],70,3).sum(axis=2)
wlr_optim = wl.reshape(70,3).mean(axis=1)
 
rmscv_min = rmscv_base # Initialise to the baseline value
iter_max= 70
for rep in range(iter_max):
    rmscv = []
    r2 = []
    loo = LeaveOneOut()
   
    for train_wl, test_wl in loo.split(wlr_optim):
        opt_comp, rmsecv_min = pls_optimise_components(Xr_optim[:,train_wl], y, 15)
        predicted, r2cv_loo, rmscv_loo = base_pls_cv(Xr_optim[:,train_wl], y, opt_comp)
        rmscv.append(rmscv_loo)
        r2.append(r2cv_loo)
        
        stdout.write("\r"+str(test_wl))
        stdout.write(" ")
        stdout.write(" %1.4f" %rmscv_loo)
        stdout.write(" ")
        stdout.write(" %1.4f" %r2cv_loo)
        stdout.write(" ")
        stdout.flush()
    
    new_rmscv = np.min(np.array(rmscv))
    
    stdout.write('\r')
    
    #print(rep, np.argmin(np.array(rmscv)), np.min(np.array(rmscv)), np.array(r2)[np.argmin(np.array(rmscv))] )    
    print("Rep: %1d,  Deleted band: %1d, RMSCV: %1.4f, R^2: %1.4f " \
        %(rep,np.argmin(np.array(rmscv)), np.min(np.array(rmscv)),np.array(r2)[np.argmin(np.array(rmscv))]) )
    if (new_rmscv < rmscv_min):
        rmscv_min = new_rmscv
        Xr_optim = np.delete(Xr_optim, np.argmin(np.array(rmscv)), axis=1)
        wlr_optim = np.delete(wlr_optim, np.argmin(np.array(rmscv)))
    else:
        print("End of optimisation at step ", rep)
        break

opt_comp, rmsecv_min = pls_optimise_components(Xr_optim, y, 20)
predicted, r2cv_optim, rmscv_optim = base_pls_cv(Xr_optim, y, opt_comp)
print("PLS results:")
print("   R2: %5.3f, RMSE: %5.3f"%(r2cv_optim, rmscv_optim))
print("   Number of Latent Variables: "+str(opt_comp))
 
regression_plot(y, predicted, title="Optimised-bands PLS", variable = "TOC (g/kg)")