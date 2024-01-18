
#average value functions
def avg_2(df, col1, col2, avg_col): #avg col1 and col2 into a new avg_col for a df
    df_Temp = df[[col1, col2]].copy()
    df[avg_col] = df_Temp.mean(axis=1)
    
def avg_3(df, col1, col2, col3, avg_col): #avg col1 col 2and col3 into a new avg_col for a df
    df_Temp = df[[col1, col2, col3]].copy()
    df[avg_col] = df_Temp.mean(axis=1)
    
#stats from Kai Fan
#Fractional Bias - FB
def fb(df,name_var1,name_var2):  #var1 is model var2 is observed
    df_new=pd.DataFrame()
    df_new[name_var1]=df[name_var1]
    df_new[name_var2]=df[name_var2]
    df_new['dif_var']=df_new[name_var1]-df_new[name_var2]
    df_new['sum_var']=df_new[name_var1]+df_new[name_var2]
    FB=round((df_new['dif_var']/df_new['sum_var']).sum()*2/len(df[name_var1])*100,1)
    return FB

#Fractional Error - FE
def fe(df,name_var1,name_var2):  #var1 is model var2 is observed
    df_new=pd.DataFrame()
    df_new[name_var1]=df[name_var1]
    df_new[name_var2]=df[name_var2]
    df_new['dif_var']= abs(df_new[name_var1]-df_new[name_var2])
    df_new['sum_var']=df_new[name_var1]+df_new[name_var2]
    FE=round((df_new['dif_var']/df_new['sum_var']).sum()*2/len(df[name_var1])*100,1)
    return FE

#Normalized Mean Bias - NMB
def nmb(df,name_var1,name_var2):  #var1 is model var2 is observed
    df_new=pd.DataFrame()
    df_new[name_var1]=df[name_var1]
    df_new[name_var2]=df[name_var2]
    df_new['dif_var']=df_new[name_var1]-df_new[name_var2]
    NMB=round((df_new['dif_var'].sum()/df_new[name_var2].sum())*100,1)
    return NMB

#Normalized Mean Error - NME
def nme(df,name_var1,name_var2):  #var1 is model var2 is observed
    df_new=pd.DataFrame()
    df_new[name_var1]=df[name_var1]
    df_new[name_var2]=df[name_var2]
    df_new['dif_var']= abs(df_new[name_var1]-df_new[name_var2])
    NME=round((df_new['dif_var'].sum()/df_new[name_var2].sum())*100,1)
    return NME

#Root Mean Squared Error - RMSE
def rmse(df,name_var1,name_var2):  #var1 is model var2 is observed
    df_new=pd.DataFrame()
    df_new[name_var1]=df[name_var1]
    df_new[name_var2]=df[name_var2]
    df_new['dif_var']= (df_new[name_var1]-df_new[name_var2])**(2)
    RMSE=round((df_new['dif_var'].sum()/len(df_new.index))**(0.5),1)
    return RMSE

#Coefficient of Determination - r^2
def r2(df,name_var1,name_var2):
    df_new=pd.DataFrame()
    df_new[name_var1]=df[name_var1]
    df_new[name_var2]=df[name_var2]
    top_var= ((df_new[name_var1]-np.mean(df_new[name_var1])) * (df_new[name_var2]-np.mean(df_new[name_var2]))).sum()
    bot_var= (((df_new[name_var1]-np.mean(df_new[name_var1]))**2).sum() * ((df_new[name_var2]-np.mean(df_new[name_var2]))**2).sum())**(.5)
    r_squared=round(((top_var/bot_var)**2),2)
    return r_squared

#combines previous stats
def stats(df,name_var1,name_var2):
    FB = fb(df,name_var1,name_var2)
    FE = fe(df,name_var1,name_var2)
    NMB = nmb(df,name_var1,name_var2)
    NME = nme(df,name_var1,name_var2)
    RMSE = rmse(df,name_var1,name_var2)
    r_squared = r2(df,name_var1,name_var2)
    g = pd.DataFrame([FB,FE,NMB,NME,RMSE,r_squared])
    g.index = ["FB","FE","NMB", "NME", "RMSE", "r_squared"]
    g.columns = [name_var1]
    return g   

def DenGraphWithStats(fig, test, pred, graphlabel, i):
    #make data frame
    dfT = pd.DataFrame({str(test): test, str(pred): pred}, columns=[str(test), str(pred)])
    
    #density graph
    x = test
    y = pred
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    ax = fig.add_subplot(1,3,i)
    ax.scatter(x, y, c=z, s=20, edgecolor='', label=graphlabel)
    ax.set_xlabel('Observed Ozone (ppb)', fontsize=10)
    ax.set_ylabel('Model Ozone Prediction (ppb)', fontsize=10)
    ax.legend(loc=2, prop={'size': 10})
    q = np.linspace(0, 100, 100)
   
    j = q
    ax.plot(q,j, 'r')
    
    #create anchored text with stats on plot
    #statslabels = AnchoredText('$R^2: $' + str(r2(dfT, str(pred), str(test))) + ' NMB_high(%): ' + str(nmb(dfT[dfT[str(test)]>54], str(pred), str(test))) + '\nNMB_low(%): ' + str(nmb(dfT[dfT[str(test)]<=54], str(pred), str(test))) + '\nNMB(%): ' + str(nmb(dfT, str(pred), str(test))) + ' NME(%): ' + str(nme(dfT, str(pred), str(test))), loc=4)                 #('\n$R^2$:' + %(.:3f)%.format(r_squared), 4)
    statslabels = AnchoredText('$R^2: $' + str(r2(dfT, str(pred), str(test))) + '\nNMB: ' + str(nmb(dfT, str(pred), str(test))) + '%\nNME: ' + str(nme(dfT, str(pred), str(test))) + '%', loc=4)  
    ax.set_ylim(0, 100)
    ax.set_xlim(0, 100)
    ax.add_artist(statslabels)
    #plt.savefig(r'C:\Users\Ryan\LARREU_18\AIRPACT-ML-master\ML_documents_images\images\Poster.png', dpi=600)
    #plt.show()
    
    
def PT_region_stats(test, pred):  
    #creates a table of stats for the test and pred data
    #create data frame for stats by range
    dfT = pd.DataFrame({str(test): test, str(pred): pred}, columns=[str(test), str(pred)])
    
    #partition into different categories 
    df40 = dfT[(dfT[str(test)] < 40)]
    df60 = dfT[(dfT[str(test)] < 60) & (dfT[str(test)] >= 40)]
    df70 = dfT[(dfT[str(test)] < 70) & (dfT[str(test)] >= 60)]
    dfinf = dfT[(dfT[str(test)] >= 70)]
    
    #create prettytable for stats 
    from prettytable import PrettyTable
    from prettytable import MSWORD_FRIENDLY
    statsT = PrettyTable()
    statsT.field_names = ["Statistic", "0-40", "40-60", "60-70", ">70"]
    statsT.add_row(["N", len(df40.index), len(df60.index), len(df70.index), len(dfinf.index)])
    statsT.add_row(["Observed Avg(ppb)", round(df40[str(test)].sum()/len(df40.index),2), round(df60[str(test)].sum()/len(df60.index),2), round(df70[str(test)].sum()/len(df70.index),2), round(dfinf[str(test)].sum()/len(dfinf.index),2)])
    statsT.add_row(["Model Avg(ppb)", round(df40[str(pred)].sum()/len(df40.index),2), round(df60[str(pred)].sum()/len(df60.index),2), round(df70[str(pred)].sum()/len(df70.index),2), round(dfinf[str(pred)].sum()/len(dfinf.index),2)])
    statsT.add_row(["RMSE(ppb)", rmse(df40, str(pred), str(test)), rmse(df60, str(pred), str(test)), rmse(df70, str(pred), str(test)), rmse(dfinf, str(pred), str(test))])
    statsT.add_row(['R^2', r2(df40, str(pred), str(test)), r2(df60, str(pred), str(test)), r2(df70, str(pred), str(test)), r2(dfinf, str(pred), str(test))])
    statsT.add_row(["NMB(%)", nmb(df40, str(pred), str(test)), nmb(df60, str(pred), str(test)), nmb(df70, str(pred), str(test)), nmb(dfinf, str(pred), str(test))])
    statsT.add_row(["NME(%)", nme(df40, str(pred), str(test)), nme(df60, str(pred), str(test)), nme(df70, str(pred), str(test)), nme(dfinf, str(pred), str(test))])
    statsT.set_style(MSWORD_FRIENDLY)
    print(statsT)
    
def PT_model_comp(observed, model):  
    #creates a table of stats for the test and pred data
    #create data frame for comparison
    #for arrays
    dfC = pd.DataFrame({str('observed'): observed, str('model'): model}, columns=[str('observed'), str('model')])
    
    #set the Ozone seperatation
    Osep = 54.5
    
    #numbers refer to where they wil be placed like cordinates starting in bottom left. first number goes left then second is right
    df11 = dfC[(dfC[str('observed')] < Osep) & (dfC[str('model')] > Osep)]   
    df22 = dfC[(dfC[str('observed')] > Osep) & (dfC[str('model')] < Osep)] 
    df12 = dfC[(dfC[str('observed')] < Osep) & (dfC[str('model')] < Osep)] 
    df21 = dfC[(dfC[str('observed')] > Osep) & (dfC[str('model')] > Osep)] 
    
    from prettytable import PrettyTable
    from prettytable import MSWORD_FRIENDLY
    ozone_region = PrettyTable()
    ozone_region.field_names = ["", "Observed < "+ str(Osep), "Observered > " + str(Osep)]
    ozone_region.add_row(["Model < " + str(Osep), len(df12), len(df22)])
    ozone_region.add_row(["Model > " + str(Osep), len(df11), len(df21)])
    ozone_region.set_style(MSWORD_FRIENDLY)
    print(ozone_region)
    
def PT_model_comp_df(observed, model, df):  
    #creates a table of stats for the test and pred data
    #create data frame for comparison
    #for dataframes

    #set the Ozone seperatation
    Osep = 54
    
    #numbers refer to where they wil be placed like cordinates starting in bottom left. first number goes left then second is right
    df11 = df[(df[observed] < Osep) & (df[model] > Osep)]   
    df22 = df[(df[observed] > Osep) & (df[model] < Osep)] 
    df12 = df[(df[observed] < Osep) & (df[model] < Osep)] 
    df21 = df[(df[observed] > Osep) & (df[model] > Osep)] 
    
    from prettytable import PrettyTable
    from prettytable import MSWORD_FRIENDLY
    ozone_region = PrettyTable()
    ozone_region.field_names = ["", "Observed < "+ str(Osep), "Observered > " + str(Osep)]
    ozone_region.add_row(["Model < " + str(Osep), len(df12), len(df22)])
    ozone_region.add_row(["Model > " + str(Osep), len(df11), len(df21)])
    ozone_region.set_style(MSWORD_FRIENDLY)
    print(ozone_region)    

def AQI_df(observed, model, df):  
    #creates a table of stats for the test and pred data
    #create data frame for comparison
    #for dataframes
    
    ozone_region  = pd.DataFrame(columns=["", 'Obs_1', 'Obs_2', 'Obs_3', 'Obs_4', 'Obs_5', 'Obs_6'])
    
    #numbers refer to where they wil be placed like cordinates starting in bottom left. first number goes left then second is right
    for m in range(1,7):   
        temp = []
        for o in range(1,7):
            temp.append(len(df[(df[observed] == o) & (df[model] == m)]))
        ozone_region.loc[m]=['Mod_'+str(m)] + temp
        
    return ozone_region

def preprocess(preprocesstype, var):
    #preprocesstype: selects preproccesing type for model, "MMS" for MinMaxScaler, "RS" for Robustscaler, "SS" for StandardScaler, "MAS" for MaxAbsScaler
    #var for varibale np.array is set to
    
    if preprocesstype == "MMS":
        print("preprocessing is done with MinMaxScaler")
        X = preprocessing.StandardScaler()
        var = X.fit_transform(var)
        return var
    elif preprocesstype == "RS":
        print("preprocessing is done with RobustScaler")
        X = preprocessing.RobustScaler()
        var = X.fit_transform(var)
        return var
    elif preprocesstype == "SS":
        print("preprocessing is done with StandardScaler")
        X = preprocessing.StandardScaler()
        var = X.fit_transform(var)
        return var
    elif preprocesstype == "MAS":
        print("preprocessing is done with MaxAbsScaler")
        X = preprocessing.MaxAbsScaler()
        var = X.fit_transform(var)
        return var
    else:
        print("Preprocessing type not recognized")
        
def feat_select(df, forecast_col, ap_col, comparisondf, preprocesstype = "MMS", f = 0): 
    
    # df - dataframe containing all feature and predictor columns
    # forecast_col - column name for the predictor   
    
    
    
    # machine learning can't take NA (or NaN in pandas)
    # Instead of getting rid of NA, we replaced them with -99999, WHICH BECOME OUTLIARS
#    df.fillna(value=-99999, inplace=True)
#    print(df.head())
    
    #added drop NA instead of adding outliers
    # get rid of NA in df - I am not sure if this is necessary
    df = df.dropna()
    
    # create new np array without label
    X = np.array(df.drop([forecast_col, ap_col], 1))
    
    X = preprocess(preprocesstype, X)
    
    # convert X to dataframe to use pairplot
    X_dat=pd.DataFrame(X)
    #X_colnames = X_dat.columns
    
    colnames = list(df.columns)
    del colnames[colnames.index(forecast_col)]
    del colnames[colnames.index(ap_col)]
    
    X_dat.columns = colnames
    print('Xdat columns', X_dat.columns)
    # pairplot of preprocessed X
#    import seaborn as sns
#    g1 = sns.pairplot(X_dat)
#    g1.savefig("preprocessed_d1.png")
    
    # separate "label" to y 
    Y = np.array(df[forecast_col])
    Z = np.array(df[ap_col])
    
    X_train, X_test, Y_train, Y_test, Z_train, Z_test= train_test_split(X, Y, Z, test_size=0.2, random_state=42)
    
    print("Mutual_info_regression feature selection starts")
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import mutual_info_regression
    
    # feature extraction
    test = SelectKBest(score_func=mutual_info_regression, k=4)
    fit = test.fit(X_train, Y_train)
#    mutualinfo_pred = test.predict(X_test)
    # summarize scores
    np.set_printoptions(precision=3)
#    print("mutualinfo score: " , fit.scores_)
    features = fit.transform(X)
    # summarize selected features
#    print(features[0:2,:])
    
    
    print("RFE, recursive feature elimination, feature selection starts")
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LinearRegression
    #from pygam import LinearGAM
    # feature extraction
    model_LR = LinearRegression()
    rfe = RFE(model_LR, n_features_to_select =  5)
    RFEfit = rfe.fit(X_train, Y_train)
    RFE_pred = RFEfit.predict(X_test)
    print("RFE: Num Features", RFEfit.n_features_ )
    print("RFE: Selected Features",  RFEfit.support_ )
#    print("RFE: Feature Ranking", RFEfit.ranking_ )
    
    print("PCA feature selection starts")
    from sklearn.decomposition import PCA
    # feature extraction
    pca = PCA(n_components=5)
    pcafit = pca.fit(X)
#    pca_pred = pcafit.predict(X_test)
    # summarize components
    print("PCA: Explained Variance", pcafit.explained_variance_ratio_ )
#    print("PCA components:" , pcafit.components_)
    
    X_pca = pca.transform(X)
    print("original shape:   ", X.shape)
    print("transformed shape:", X_pca.shape)
    X_new = pca.inverse_transform(X_pca)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
    plt.axis('equal');
    plt.show()
    
    #plt.plot(np.cumsum(pcafit.explained_variance_ratio_))
    
    print("Randome Forest regressor feature selection starts")
    from sklearn.ensemble import RandomForestRegressor
    # feature extraction
    model_RF = RandomForestRegressor(n_estimators=100, max_depth=7)
    model_RF = model_RF.fit(X_train, Y_train)
    RF_pred = model_RF.predict(X_test)
#    print("Randomforest regressor:", model_RF.feature_importances_)
    
    #gam = LinearGAM(n_splines=10).gridsearch(X_train,Y_train)
    #GAM_pred = gam.predict(X_test)
    from numpy import loadtxt
    from xgboost import XGBRegressor
    model_boost = XGBRegressor()
    model_boost.fit(X_train, Y_train)
    boost_pred = model_boost.predict(X_test)
    
    #**graphing is now in a function for a density scatter plot below not needed
    # compare all predicted values
#    plt.plot(Y_test, pca_pred, color="b")
    # quick scatter plot between original y data and predicted y data
    #lineStart = Y_test.min()*0.6 
    #lineEnd = Y_test.max()*1.05 
    
    #commented out, making the green line that isnt helping
    #plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'g')
    
    fig = plt.figure(figsize=(24,4))
    #graphs and stats for linear regression
    DenGraphWithStats(fig, Y_test, RFE_pred, "MLR Test 2015-2017", 2)
    #pretty table with stats by region for linear regression
    PT_region_stats(Y_test, RFE_pred)
    
    #graphs with stats for random forest
    DenGraphWithStats(fig, Y_test, RF_pred, "RF Test 2015-2017", 4)
    #pretty table with stats by region for random forest
    PT_region_stats(Y_test, RF_pred)
    
    Z_test = Z_test[~np.isnan(Z_test)]
    #compare airpact graph
    DenGraphWithStats(fig, Y_test, Z_test, "AIRPACT Test 2015-2017", 1)
    #pretty table with stats by region for random forest
    PT_region_stats(Y_test, Z_test)
    ##compare GAM
    #DenGraphWithStats(fig, Y_test, GAM_pred, "GAM Test 2015-2017", 3)
    ##pretty table with stats by region for random forest
    #PT_region_stats(Y_test, GAM_pred)
    #plt.tight_layout()
    #plt.show()
    #compare XGBoost
    DenGraphWithStats(fig, Y_test, boost_pred, "XGBoost Test 2015-2017", 3)
    #pretty table with stats by region for random forest
    PT_region_stats(Y_test, boost_pred)
    plt.tight_layout()
    plt.show()
    
    #Chart for comparing airpact and random forest
    print('AIRPACT comparison')
    PT_model_comp(Y_test, Z_test)
    
    print('Randomforest comparison')
    PT_model_comp(Y_test, RF_pred)
    
    print('Linear Regression comparison')
    PT_model_comp(Y_test, RFE_pred)
    
    print('GAM Regression comparison')
    PT_model_comp(Y_test, boost_pred)
    
    # make a pretty table for variables
    from prettytable import PrettyTable
    from prettytable import MSWORD_FRIENDLY
    tab = PrettyTable()
    tab.add_column("feature_sel", colnames )
    tab.add_column("Mutual_info_score", ["%.3f" % member for member in fit.scores_] ) 
    tab.add_column("RFE: Feature Ranking", RFEfit.ranking_ ) 
    tab.add_column("PCA_comps_0", ["%.3f" % member for member in pcafit.components_[0] ] )
    tab.add_column("PCA_comps_1",["%.3f" % member for member in pcafit.components_[1] ]  )
    tab.add_column("PCA_comps_2",["%.3f" % member for member in pcafit.components_[2] ]  )
    tab.add_column("RF_feature_import",["%.3f" % member for member in model_RF.feature_importances_ ])   
    tab.add_column("XGBoost_feature_import",["%.3f" % member for member in model_boost.feature_importances_ ])       
    tab.set_style(MSWORD_FRIENDLY)
    print(tab)

    
    # plot random forest tree
    # the dot file created below can't be opened, so please don't run the code below. 
    #import graphviz
    #from sklearn import tree
    #dotfile = open("./randomforest_tree0.dot", 'w')
    #dot_data = tree.export_graphviz(model_RF.estimators_[0], out_file = dotfile)
    #dotfile.close()
    #graph.render(X)
    
        # machine learning can't take NA (or NaN in pandas)
    # Instead of getting rid of NA, we replaced them with -99999, WHICH BECOME OUTLIARS
#    df.fillna(value=-99999, inplace=True)
#    print(df.head())
    
    #

    #added drop NA instead of adding outlier
    comparisondf = comparisondf.dropna()
    # create new np array without labels
    M = np.array(comparisondf.drop([forecast_col, ap_col], 1))
    
    M = preprocess(preprocesstype, M)
    
    # convert M to dataframe to use pairplot
    M_dat=pd.DataFrame(M)
    #M_colnames = M_dat.columns
    
    colnamescomp = list(comparisondf.columns)
    del colnamescomp[colnamescomp.index(forecast_col)]
    del colnamescomp[colnamescomp.index(ap_col)]
    
    M_dat.columns = colnamescomp
    print('Mdat columns', M_dat.columns)
    # pairplot of preprocessed X
#        import seaborn as sns
#        g1 = sns.pairplot(M_dat)
#        g1.savefig("preprocessed_d1.png")
    
    # separate "label" to N 
    N = np.array(comparisondf[forecast_col])
    L = np.array(comparisondf[ap_col])

    #linear regression prediction
    RFE_predcomp = RFEfit.predict(M)

    #random forest prediction
    RF_predcomp = model_RF.predict(M)
    
    ##gam prediction
    #GAM_predcomp = gam.predict(M)
    #XGBoost prediction
    boost_predcomp = model_boost.predict(M)
    
    fig = plt.figure(figsize=(24,4))
    #linear regression
    #graphs with stats for linear regression
    DenGraphWithStats(fig, N, RFE_predcomp, "MLR 2018", 2)
    #pretty table with stats by region for linear regression
    PT_region_stats(N, RFE_predcomp)
    
    #random forest
    #graphs with stats for random forest
    DenGraphWithStats(fig, N, RF_predcomp, "RF 2018", 4)
    #pretty table with stats by region for random forest
    PT_region_stats(N, RF_predcomp)
    
    #airpact
    #graphs with stats for random forest
    DenGraphWithStats(fig, N, L, "AIRPACT 2018", 1)
    #pretty table with stats by region for random forest
    PT_region_stats(N, L)
    
    #gam
    #graphs with stats for random forest
    DenGraphWithStats(fig, N, boost_predcomp, "XGBoost 2018", 3)
    #pretty table with stats by region for random forest
    PT_region_stats(N, boost_predcomp)
    #plt.savefig(r'/Users/fankai/Downloads/1.png', dpi=600)
    
    #import seaborn as sns
    #from scipy import stats
    #ax = fig.add_subplot(1,5,5)
    #sns.distplot(RF_predcomp)
    #sns.distplot(comparisondf[forecast_col])
    plt.tight_layout()
    plt.show()
    
    #make dataframe for comparison
    dfO3 = comparisondf[[forecast_col, ap_col]].copy()
    dfO3['RF']= RF_predcomp
    dfO3['RFE']= RFE_predcomp
    #dfO3['GAM']= GAM_predcomp
    dfO3['XGBoost']= boost_predcomp

    #calculate rolling average of 8 hours then move them up since function calculates back not forward
    #calculate 8 hour average
    dfO3['AIRPACT.avg8hr'] = dfO3[ap_col].rolling(8, min_periods=6).mean()
    dfO3['Observed.avg8hr'] = dfO3[forecast_col].rolling(8, min_periods=6).mean()
    dfO3['Randomforest.avg8hr'] = dfO3['RF'].rolling(8, min_periods=6).mean()
    dfO3['Linearregression.avg8hr'] = dfO3['RFE'].rolling(8, min_periods=6).mean()
    #dfO3['GAM.avg8hr'] = dfO3['GAM'].rolling(8, min_periods=6).mean()
    dfO3['boost.avg8hr'] = dfO3['XGBoost'].rolling(8, min_periods=6).mean()
    
    #shift columns
    dfO3['AIRPACT.avg8hr'] = dfO3['AIRPACT.avg8hr'].shift(-7)
    dfO3['Observed.avg8hr'] = dfO3['Observed.avg8hr'].shift(-7)
    dfO3['Randomforest.avg8hr'] = dfO3['Randomforest.avg8hr'].shift(-7)
    dfO3['Linearregression.avg8hr'] = dfO3['Linearregression.avg8hr'].shift(-7)
    #dfO3['GAM.avg8hr'] = dfO3['GAM.avg8hr'].shift(-7).
    dfO3['boost.avg8hr'] = dfO3['boost.avg8hr'].shift(-7)
    
    #calculate the highest 8 hour avg for every day this will leave the correct value at 7am every day
    dfO3['AP.maxdaily8hravg'] = dfO3['AIRPACT.avg8hr'].rolling(17, min_periods=13).max()
    dfO3['O.maxdaily8hravg'] = dfO3['Observed.avg8hr'].rolling(17, min_periods=13).max()
    dfO3['RF.maxdaily8hravg'] = dfO3['Randomforest.avg8hr'].rolling(17, min_periods=13).max()
    dfO3['LR.maxdaily8hravg'] = dfO3['Linearregression.avg8hr'].rolling(17, min_periods=13).max()
    #dfO3['GAM.maxdaily8hravg'] = dfO3['GAM.avg8hr'].rolling(17, min_periods=13).max()
    dfO3['boost.maxdaily8hravg'] = dfO3['boost.avg8hr'].rolling(17, min_periods=13).max()
    
    #shift columns
    dfO3['AP.maxdaily8hravg'] = dfO3['AP.maxdaily8hravg'].shift(-16)
    dfO3['O.maxdaily8hravg'] = dfO3['O.maxdaily8hravg'].shift(-16)   
    dfO3['RF.maxdaily8hravg'] = dfO3['RF.maxdaily8hravg'].shift(-16)
    dfO3['LR.maxdaily8hravg'] = dfO3['LR.maxdaily8hravg'].shift(-16)
    #dfO3['GAM.maxdaily8hravg'] = dfO3['GAM.maxdaily8hravg'].shift(-16)
    dfO3['boost.maxdaily8hravg'] = dfO3['boost.maxdaily8hravg'].shift(-16)
    #return dfO3
    #pull out the 7am value which represents the 8 hour average daily maximum
    dfdailyO38hrmax = dfO3[(dfO3.index.hour == 7)]
    
    #repeat process used for 8hr avg above but for daily max instead
    daily8hrO3max=dfdailyO38hrmax.dropna()
    
    if(f=='full'):
        fig = plt.figure(figsize=(24,4))
        #linear regression
        #graphs with stats for linear regression
        DenGraphWithStats(fig, daily8hrO3max['O.maxdaily8hravg'], daily8hrO3max['LR.maxdaily8hravg'], "LR daily 8hr max 2018", 2)
        q = np.linspace(54, 54, 100)
        j = np.linspace(0, 100, 100)
        plt.plot(q,j, 'b')
        plt.plot(j,q, 'b')
        #pretty table with stats by region for linear regression
        PT_region_stats(daily8hrO3max['O.maxdaily8hravg'], daily8hrO3max['LR.maxdaily8hravg'])
        
        #random forest
        #graphs with stats for random forest
        DenGraphWithStats(fig, daily8hrO3max['O.maxdaily8hravg'], daily8hrO3max['RF.maxdaily8hravg'], "RF daily 8hr max 2018",4)
        q = np.linspace(54, 54, 100)
        j = np.linspace(0, 100, 100)
        plt.plot(q,j, 'b')
        plt.plot(j,q, 'b')
        #pretty table with stats by region for random forest
        PT_region_stats(daily8hrO3max['O.maxdaily8hravg'], daily8hrO3max['RF.maxdaily8hravg'])
        
        #airpact
        #graphs with stats for random forest
        DenGraphWithStats(fig, daily8hrO3max['O.maxdaily8hravg'], daily8hrO3max['AP.maxdaily8hravg'], "AIRPACT daily 8hr max 2018", 1)
        q = np.linspace(54, 54, 100)
        j = np.linspace(0, 100, 100)
        plt.plot(q,j, 'b')
        plt.plot(j,q, 'b')
        #pretty table with stats by region for random forest
        PT_region_stats(daily8hrO3max['O.maxdaily8hravg'], daily8hrO3max['AP.maxdaily8hravg'])
        
        ##GAM
        ##graphs with stats for GAM
        #DenGraphWithStats(fig, daily8hrO3max['O.maxdaily8hravg'], daily8hrO3max['GAM.maxdaily8hravg'], "GAM daily 8hr max 2018", 3)
        #q = np.linspace(54, 54, 100)
        #j = np.linspace(0, 100, 100)
        #plt.plot(q,j, 'b')
        #plt.plot(j,q, 'b')
        ##pretty table with stats by region for GAM
        #PT_region_stats(daily8hrO3max['O.maxdaily8hravg'], daily8hrO3max['GAM.maxdaily8hravg'])
        #plt.tight_layout()
        #plt.show()
        #XGBoost
        #graphs with stats for GAM
        DenGraphWithStats(fig, daily8hrO3max['O.maxdaily8hravg'], daily8hrO3max['boost.maxdaily8hravg'], "XGBoost daily 8hr max 2018", 3)
        q = np.linspace(54, 54, 100)
        j = np.linspace(0, 100, 100)
        plt.plot(q,j, 'b')
        plt.plot(j,q, 'b')
        #pretty table with stats by region for XGBoost
        PT_region_stats(daily8hrO3max['O.maxdaily8hravg'], daily8hrO3max['boost.maxdaily8hravg'])
        plt.tight_layout()
        plt.show()
    
    #do comparison for airpact
    print('AIRPACT')
    PT_model_comp_df('O.maxdaily8hravg', 'AP.maxdaily8hravg', dfdailyO38hrmax)
    print('Random Forest')
    PT_model_comp_df('O.maxdaily8hravg', 'RF.maxdaily8hravg', dfdailyO38hrmax)
    print('Linear Regression')
    PT_model_comp_df('O.maxdaily8hravg', 'LR.maxdaily8hravg', dfdailyO38hrmax)
    #print('GAM')
    #PT_model_comp_df('O.maxdaily8hravg', 'GAM.maxdaily8hravg', dfdailyO38hrmax)
    print('XGBoost')
    PT_model_comp_df('O.maxdaily8hravg', 'boost.maxdaily8hravg', dfdailyO38hrmax)


def feat_selectallPP(df, forecast_col, comparisondf = None):
    #takes model function and runs it with all four preprocessing types
    feat_select(df, forecast_col, "MMS", comparisondf)
    feat_select(df, forecast_col, "RS", comparisondf)
    feat_select(df, forecast_col, "SS", comparisondf)
    feat_select(df, forecast_col, "MAS", comparisondf)
    

def feat_select_class(df, forecast_col, ap_col, comparisondf, preprocesstype = "MMS"): 
    
    # df - dataframe containing all feature and predictor columns
    # forecast_col - column name for the predictor   
    
    
    
    # machine learning can't take NA (or NaN in pandas)
    # Instead of getting rid of NA, we replaced them with -99999, WHICH BECOME OUTLIARS
#    df.fillna(value=-99999, inplace=True)
#    print(df.head())
    
    #added drop NA instead of adding outliers
    # get rid of NA in df - I am not sure if this is necessary
    df = df.dropna()

    # create new np array without label
    X = np.array(df.drop([forecast_col, ap_col], 1))
    
    X = preprocess(preprocesstype, X)
    
    # convert X to dataframe to use pairplot
    X_dat=pd.DataFrame(X)
    #X_colnames = X_dat.columns
    
    colnames = list(df.columns)
    del colnames[colnames.index(forecast_col)]
    del colnames[colnames.index(ap_col)]
    
    X_dat.columns = colnames
    print('Xdat columns', X_dat.columns)
    # pairplot of preprocessed X
#    import seaborn as sns
#    g1 = sns.pairplot(X_dat)
#    g1.savefig("preprocessed_d1.png")
    
    # separate "label" to y 
    Y = np.array(df[forecast_col])
    Z = np.array(df[ap_col])
    
    X_train, X_test, Y_train, Y_test, Z_train, Z_test= train_test_split(X, Y, Z, test_size=0.2, random_state=42)

    print("Mutual_info_regression feature selection starts")
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import mutual_info_regression
    
    # feature extraction
    test = SelectKBest(score_func=mutual_info_regression, k=4)
    fit = test.fit(X_train, Y_train)
#    mutualinfo_pred = test.predict(X_test)
    # summarize scores
    np.set_printoptions(precision=3)
#    print("mutualinfo score: " , fit.scores_)
    features = fit.transform(X)
    # summarize selected features
#    print(features[0:2,:])
   
    print("Randome Forest regressor feature selection starts")
    from sklearn.ensemble import RandomForestClassifier
    # feature extraction
    model_RF = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt',
                               class_weight = dict({0:30, 1:1}))
    model_RF = model_RF.fit(X_train, Y_train)
    RF_pred = model_RF.predict(X_test)
#    print("Randomforest regressor:", model_RF.feature_importances_)
    
    #Chart for comparing airpact and random forest
    print('AIRPACT comparison')
    PT_model_comp(Y_test, Z_test)
    
    print('Randomforest comparison')
    PT_model_comp(Y_test, RF_pred)
    
    # make a pretty table for variables
    from prettytable import PrettyTable
    from prettytable import MSWORD_FRIENDLY
    tab = PrettyTable()
    tab.add_column("feature_sel", colnames )
    tab.add_column("RF_feature_import",["%.3f" % member for member in model_RF.feature_importances_ ])     
    tab.set_style(MSWORD_FRIENDLY)
    print(tab)

    
    # plot random forest tree
    # the dot file created below can't be opened, so please don't run the code below. 
    #import graphviz
    #from sklearn import tree
    #dotfile = open("./randomforest_tree0.dot", 'w')
    #dot_data = tree.export_graphviz(model_RF.estimators_[0], out_file = dotfile)
    #dotfile.close()
    #graph.render(X)
    
        # machine learning can't take NA (or NaN in pandas)
    # Instead of getting rid of NA, we replaced them with -99999, WHICH BECOME OUTLIARS
#    df.fillna(value=-99999, inplace=True)
#    print(df.head())
    
    #

    #added drop NA instead of adding outlier
    comparisondf = comparisondf.dropna()
    # create new np array without labels
    M = np.array(comparisondf.drop([forecast_col, ap_col], 1))
    
    M = preprocess(preprocesstype, M)
    
    # convert M to dataframe to use pairplot
    M_dat=pd.DataFrame(M)
    #M_colnames = M_dat.columns
    
    colnamescomp = list(comparisondf.columns)
    del colnamescomp[colnamescomp.index(forecast_col)]
    del colnamescomp[colnamescomp.index(ap_col)]
    
    M_dat.columns = colnamescomp
    print('Mdat columns', M_dat.columns)
    # pairplot of preprocessed X
#        import seaborn as sns
#        g1 = sns.pairplot(M_dat)
#        g1.savefig("preprocessed_d1.png")
    
    # separate "label" to N 
    N = np.array(comparisondf[forecast_col])
    L = np.array(comparisondf[ap_col])

    #random forest prediction
    RF_predcomp = model_RF.predict(M)
    
    #make dataframe for comparison
    dfO3 = comparisondf[[forecast_col, ap_col]].copy()
    dfO3['RF']= RF_predcomp

    #calculate rolling average of 8 hours then move them up since function calculates back not forward
    
    #calculate the highest 8 hour avg for every day this will leave the correct value at 7am every day
    #shift columns
    dfO3['AP.maxdaily8hravg'] = dfO3[ap_col].shift(-16)
    dfO3['O.maxdaily8hravg'] = dfO3[forecast_col].shift(-16)   
    dfO3['RF.maxdaily8hravg'] = dfO3['RF'].shift(-16)
    #return dfO3
    #pull out the 7am value which represents the 8 hour average daily maximum
    dfdailyO38hrmax = dfO3[dfO3.index.hour > 6]
    enough_points = dfdailyO38hrmax.resample("D").count() >= 13
    dfdailyO38hrmax = dfdailyO38hrmax.resample('D', how='max')
    dfdailyO38hrmax = dfdailyO38hrmax[enough_points]
    
    #do comparison for airpact
    print('AIRPACT')
    PT_model_comp_df('O.maxdaily8hravg', 'AP.maxdaily8hravg', dfdailyO38hrmax)
    print('Random Forest')
    PT_model_comp_df('O.maxdaily8hravg', 'RF.maxdaily8hravg', dfdailyO38hrmax)


def feat_elimination(df, forecast_col, ap_col, comparisondf, preprocesstype = "MMS"): 
    
    # df - dataframe containing all feature and predictor columns
    # forecast_col - column name for the predictor
    # machine learning can't take NA (or NaN in pandas)
    # Instead of getting rid of NA, we replaced them with -99999, WHICH BECOME OUTLIARS
#    df.fillna(value=-99999, inplace=True)
#    print(df.head())
    
    #added drop NA instead of adding outliers
    # get rid of NA in df - I am not sure if this is necessary
    df = df.dropna()
    
    # create new np array without label
    X = np.array(df.drop([forecast_col, ap_col], 1))
    
    X = preprocess(preprocesstype, X)
    
    # convert X to dataframe to use pairplot
    X_dat=pd.DataFrame(X)
    #X_colnames = X_dat.columns
    
    colnames = list(df.columns)
    del colnames[colnames.index(forecast_col)]
    del colnames[colnames.index(ap_col)]
    
    X_dat.columns = colnames
    # pairplot of preprocessed X
#    import seaborn as sns
#    g1 = sns.pairplot(X_dat)
#    g1.savefig("preprocessed_d1.png")
    
    # separate "label" to y 
    Y = np.array(df[forecast_col])
    Z = np.array(df[ap_col])
    
    X_train, X_test, Y_train, Y_test, Z_train, Z_test= train_test_split(X, Y, Z, test_size=0.2, random_state=42)
    
    
    # feature extraction
    from sklearn.ensemble import RandomForestRegressor
    # feature extraction
    model_RF = RandomForestRegressor(n_estimators=100, max_depth=7)
    model_RF = model_RF.fit(X_train, Y_train)
    
    #added drop NA instead of adding outlier
    comparisondf = comparisondf.dropna()
    # create new np array without labels
    M = np.array(comparisondf.drop([forecast_col, ap_col], 1))
    
    M = preprocess(preprocesstype, M)
    
    # convert M to dataframe to use pairplot
    M_dat=pd.DataFrame(M)
    #M_colnames = M_dat.columns
    
    colnamescomp = list(comparisondf.columns)
    del colnamescomp[colnamescomp.index(forecast_col)]
    del colnamescomp[colnamescomp.index(ap_col)]
    
    M_dat.columns = colnamescomp
    # pairplot of preprocessed X
#        import seaborn as sns
#        g1 = sns.pairplot(M_dat)
#        g1.savefig("preprocessed_d1.png")
    
    # separate "label" to N 
    N = np.array(comparisondf[forecast_col])
    L = np.array(comparisondf[ap_col])
    #random forest prediction
    RF_predcomp = model_RF.predict(M)
    
    #random forest
    #make dataframe for comparison
    dfO3 = comparisondf[[forecast_col, ap_col]].copy()
    dfO3['RF']= RF_predcomp
    #calculate rolling average of 8 hours then move them up since function calculates back not forward
    #calculate 8 hour average
    '''
    dfO3['Observed.avg8hr'] = dfO3[forecast_col].rolling(8, min_periods=6).mean()
    dfO3['Randomforest.avg8hr'] = dfO3['RF'].rolling(8, min_periods=6).mean()
    
    #shift columns
    dfO3['Observed.avg8hr'] = dfO3['Observed.avg8hr'].shift(-7)
    dfO3['Randomforest.avg8hr'] = dfO3['Randomforest.avg8hr'].shift(-7)
    
    
    #calculate the highest 8 hour avg for every day this will leave the correct value at 7am every day
    
    dfO3['O.maxdaily8hravg'] = dfO3['Observed.avg8hr'].rolling(17, min_periods=13).max()
    dfO3['RF.maxdaily8hravg'] = dfO3['Randomforest.avg8hr'].rolling(17, min_periods=13).max()
    
    
    #shift columns
    
    dfO3['O.maxdaily8hravg'] = dfO3['O.maxdaily8hravg'].shift(-16)   
    dfO3['RF.maxdaily8hravg'] = dfO3['RF.maxdaily8hravg'].shift(-16)
    #return dfO3
    #pull out the 7am value which represents the 8 hour average daily maximum
    dfdailyO38hrmax = dfO3[(dfO3.index.hour == 7)]
    
    #repeat process used for 8hr avg above but for daily max instead
    daily8hrO3max=dfdailyO38hrmax.dropna()
    high_daily = daily8hrO3max[daily8hrO3max['O.maxdaily8hravg']>50]
    right_high = len(high_daily[high_daily['RF.maxdaily8hravg']>50])/len(high_daily)*100
    '''
    high_daily = dfO3[dfO3[forecast_col]>50]
    right_high = len(high_daily[high_daily['RF']>50])/len(high_daily)*100
    return right_high
