import json
import csv
import os
import glob
import random
from flask import Flask, render_template, request, redirect, Response, jsonify, make_response
import pandas as pd
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 
import numpy as np 
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn import manifold
import seaborn as sns
sns.set(style="ticks")

app = Flask(__name__)
randomSampleCount = int(0.25 * 1000)
feature_selected = ''
categorical_features = ['Type','Year','City','Rating','Major Size','Population']
numerical_features = ['Price','Total Volume','Total Bags','PLU4046','PLU4225','PLU4770','Small Bags','Large Bags','XLarge Bags']
stratified_sample = pd.DataFrame()
random_sample = pd.DataFrame()
eigen_values_org = []
eigen_values_random = []
eigen_values_stratified = []  
eigen_vectors_org = []
eigen_vectors_random = []
eigen_vectors_stratified = []
x_ticks_sqLoad_org = []
x_ticks_sqLoad_random = []
x_ticks_sqLoad_strat = []
squared_loadings_org = []
squared_loadings_random = []
squared_loadings_strat = []
colors=["red","green","blue","pink","yellow"]

y_ticks_org = []
y_ticks_random = []
y_ticks_stratified = []
clustered_sample = []


@app.route("/", methods = ['POST', 'GET'])
def index():
    return render_template("index_2.html")

@app.route("/dashboard", methods = ['POST', 'GET'])
def dashboard():
    return render_template("visual.html")

@app.route("/asgmt2", methods = ['POST', 'GET'])
def asgmt2():
    return render_template("asgmt2.html")

@app.route("/view_dataset", methods = ['POST', 'GET'])
def view_dataset():
    return render_template("dataset.html")

@app.route("/get_random_sample", methods = ['POST', 'GET'])
def random_sample():
    global random_sample
    csvdata =  pd.read_csv('./static/dataset/avocado_dataset.csv', low_memory = False) 
    csvdata = csvdata.filter(numerical_features, axis=1)
    random_s = csvdata.sample(randomSampleCount)
    random_sample = pd.DataFrame(columns=csvdata.columns)
    random_sample = random_sample.append(random_s)

    resp = make_response(random_sample.to_json())
    resp.status_code = 200
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp 

@app.route("/get_elbow", methods = ['POST', 'GET'])
def get_elbow():

    stdScaler = StandardScaler()
    K = range(1,10) 
    csvdata =  pd.read_csv('./static/dataset/avocado_dataset.csv', low_memory = False) 
    csvdata = csvdata.filter(numerical_features, axis=1)
    csvdata = stdScaler.fit_transform(csvdata)

    distortions = []
    # Remove old elbow plot
    files = glob.glob('./static/image/elbow/elbow.png')
    for f in files:
        os.chmod(f, 0o777)
        if 'elbow' in f:
            os.remove(f)

    for itemK in K: 
        #Building and fitting the model 
        kmeanModel = KMeans(n_clusters=itemK).fit(csvdata) 
        distortions.append(sum(np.min(cdist(csvdata, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / csvdata.shape[0]) 

    res = pd.DataFrame(columns=["x","y"])
    res['x'] = K
    res['y'] = distortions

    plt.figure()
    plt.plot(K, distortions, 'bx-') 
    plt.xlabel('Axis Number') 
    plt.ylabel('Distortion') 
    plt.title('The Elbow Method using Distortion') 
    plt.savefig('./static/image/elbow/elbow.png') 
    
    return json.dumps(res.to_json()) 

@app.route("/get_stratified_sample", methods = ['POST', 'GET'])
def get_stratified_sample(): 
    global stratified_sample
    global clustered_sample
    
    stdScaler = StandardScaler()
    cluster_size = int(request.form['size'])
    csvdata =  pd.read_csv('./static/dataset/avocado_dataset.csv', low_memory = False) 
    csvdata = csvdata.filter(numerical_features, axis=1)

    kmeanModel = KMeans(n_clusters=cluster_size).fit(csvdata) 
    sample_size_per_cluster = int(250/cluster_size)
    stratified_sample = []
    clustered_points = {}

    for index in range(0, len(kmeanModel.labels_)):
        cluster_index = kmeanModel.labels_[index]
        arr = []
        if cluster_index in clustered_points:
            arr = clustered_points[cluster_index]

        arr.append(csvdata.iloc[index])
        clustered_points[cluster_index] = arr
 
    cols = csvdata.columns
    stratified_sample = pd.DataFrame(columns=cols)
    for index in range(0, cluster_size):
        df = pd.DataFrame(clustered_points[index]).sample(sample_size_per_cluster)
        df["clusterId"] = index
        stratified_sample = stratified_sample.append(df)

     
    clustered_sample = stratified_sample.copy()
    clustered_sample["clusterId"] = clustered_sample["clusterId"].astype(int)
    stratified_sample = stratified_sample.drop("clusterId",axis=1)
    print("cluster : \n\n",clustered_sample)
    
    resp = make_response(stratified_sample.to_json())
    resp.status_code = 200
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp 


@app.route("/get_intrinsic_dimensionality_org", methods = ['POST', 'GET'])
def get_intrinsic_dimensionality_org(): 

    global eigen_values_org
    global eigen_vectors_org

    # Remove old elbow plot
    files = glob.glob('./static/image/elbow/elbow_eigen_org.png')
    for f in files:
        os.chmod(f, 0o777)
        if 'elbow_eigen_org' in f:
            os.remove(f)

    stdScaler = StandardScaler()
    csvdata =  pd.read_csv('./static/dataset/avocado_dataset.csv', low_memory = False) 
    csvdata = csvdata.filter(numerical_features, axis=1)

    csvdata = stdScaler.fit_transform(csvdata)
    covariance_matrix = np.cov(csvdata.T)
    eigen_values_org, eigen_vectors_org = np.linalg.eig(covariance_matrix)

    eigen_values_org = np.sort(eigen_values_org)
    eigen_vectors_org = np.sort(eigen_vectors_org)

    eigen_values_org_temp = eigen_values_org[::-1]

    sum_eigen = np.sum(eigen_values_org)
    variance = []
    for index in range(0,len(eigen_values_org_temp)):
        variance.append(float(eigen_values_org_temp[index]/sum_eigen*100))

    cumulative_variance = np.cumsum(variance)
   
    K = []
    for num in range(1,len(eigen_values_org_temp) + 1):
        K.append(num)
    
    map1 = {}
    for i in range(0,len(variance)):
        map1[variance[i]] = cumulative_variance[i]
    
    variance.sort(reverse=True)
    cumulative_variance = []
    for i in range(0,len(variance)):
        cumulative_variance.append(map1[variance[i]])

    res = pd.DataFrame(columns=["x","y_bar","y_plot"])
    res['x'] = K
    res['y_bar'] = variance
    res['y_plot'] = cumulative_variance

    plt.figure()
    plt.plot(K, cumulative_variance, 'bx-')
    plt.bar(K, variance)  
    plt.xlabel('Principal Component') 
    plt.ylabel('Variance Explained (%)') 
    plt.title('PCA of 9 Avocado Variables (Original Data)') 
    plt.savefig('./static/image/elbow/elbow_eigen_org.png') 

    return json.dumps(res.to_json()) 


@app.route("/get_intrinsic_dimensionality_random", methods = ['POST', 'GET'])
def get_intrinsic_dimensionality_random(): 

    global eigen_values_random
    global eigen_vectors_random

    # Remove old elbow plot
    files = glob.glob('./static/image/elbow/elbow_eigen_random.png')
    for f in files:
        os.chmod(f, 0o777)
        if 'elbow_eigen_random' in f:
            os.remove(f)

    stdScaler = StandardScaler()
    csvdata = random_sample
    csvdata = stdScaler.fit_transform(csvdata)
    covariance_matrix = np.cov(csvdata.T)
    eigen_values_random, eigen_vectors_random = np.linalg.eig(covariance_matrix)

    eigen_values_random = np.sort(eigen_values_random)
    eigen_vectors_random = np.sort(eigen_vectors_random)

    eigen_values_random_temp = eigen_values_random[::-1]

    sum_eigen = np.sum(eigen_values_random)
    variance = []
    for index in range(0,len(eigen_values_random_temp)):
        variance.append(float(eigen_values_random_temp[index]/sum_eigen*100))

    cumulative_variance = np.cumsum(variance)

    K = []
    for num in range(1,len(eigen_values_random_temp) + 1):
        K.append(num)
        
    
    map1 = {}
    for i in range(0,len(variance)):
        map1[variance[i]] = cumulative_variance[i]
    
    variance.sort(reverse=True)
    cumulative_variance = []
    for i in range(0,len(variance)):
        cumulative_variance.append(map1[variance[i]])

    res = pd.DataFrame(columns=["x","y_bar","y_plot"])
    res['x'] = K
    res['y_bar'] = variance
    res['y_plot'] = cumulative_variance

    plt.figure()
    plt.plot(K, cumulative_variance, 'bx-') 
    plt.bar(K, variance) 
    plt.xlabel('Principal Component') 
    plt.ylabel('Variance Explained (%)') 
    plt.title('PCA of 9 Avocado Variables (Random Sample)') 
    plt.savefig('./static/image/elbow/elbow_eigen_random.png') 

    return json.dumps(res.to_json()) 


@app.route("/get_intrinsic_dimensionality_strat", methods = ['POST', 'GET'])
def get_intrinsic_dimensionality_strat(): 
     # Remove old elbow plot
    global eigen_values_stratified 
    global eigen_vectors_stratified


    files = glob.glob('./static/image/elbow/elbow_eigen_stratified.png')
    for f in files:
        os.chmod(f, 0o777)
        if 'elbow_eigen_stratified' in f:
            os.remove(f)

    stdScaler = StandardScaler()
    csvdata = stratified_sample
    csvdata = stdScaler.fit_transform(csvdata)
    covariance_matrix = np.cov(csvdata.T)
    eigen_values_stratified, eigen_vectors_stratified = np.linalg.eig(covariance_matrix)

    eigen_values_stratified = np.sort(eigen_values_stratified)
    eigen_vectors_stratified = np.sort(eigen_vectors_stratified)

    eigen_values_stratified_temp = eigen_values_stratified[::-1]

    sum_eigen = np.sum(eigen_values_stratified)
    variance = []
    for index in range(0,len(eigen_values_stratified_temp)):
        variance.append(float(eigen_values_stratified_temp[index]/sum_eigen*100))

    cumulative_variance = np.cumsum(variance)

    K = []
    for num in range(1,len(eigen_values_stratified_temp) + 1):
        K.append(num)
    
     
    map1 = {}
    for i in range(0,len(variance)):
        map1[variance[i]] = cumulative_variance[i]
    
    variance.sort(reverse=True)
    cumulative_variance = []
    for i in range(0,len(variance)):
        cumulative_variance.append(map1[variance[i]])

    res = pd.DataFrame(columns=["x","y_bar","y_plot"])
    res['x'] = K
    res['y_bar'] = variance
    res['y_plot'] = cumulative_variance

    plt.figure()
    plt.plot(K, cumulative_variance, 'bx-') 
    plt.bar(K, variance) 
    plt.xlabel('Principal Component') 
    plt.ylabel('Variance Explained (%)') 
    plt.title('PCA of 9 Avocado Variables (Stratified Data))') 
    plt.savefig('./static/image/elbow/elbow_eigen_stratified.png') 

    return json.dumps(res.to_json()) 





# Square Loadings Org
@app.route("/get_top_square_loadings_org", methods = ['POST', 'GET'])
def get_top_square_loadings_org(): 

    global x_ticks_sqLoad_org
    global y_ticks_org
    global squared_loadings_org

    files = glob.glob('./static/image/elbow/square_loading_org.png')
    for f in files:
        os.chmod(f, 0o777)
        if 'square_loading_org' in f:
            os.remove(f)

    num_PCA = int(request.form['num_PCA'])
    print("Num PCA - ",num_PCA)
    squared_loadings_org = {}
    for variable_id in range(0, len(eigen_vectors_org)):
        loadings = 0
        for PCA_Comp_Num in range(0, num_PCA):
            loadings = loadings + eigen_vectors_org[PCA_Comp_Num][variable_id] * eigen_vectors_org[PCA_Comp_Num][variable_id]
        squared_loadings_org[numerical_features[variable_id]] = loadings

    squared_loadings_org = sorted(squared_loadings_org.items(), key=lambda x: x[1], reverse=True)  

    x_ticks = []
    y_ticks = []
    for var_name, sqrd_loading in squared_loadings_org:
        x_ticks.append(var_name)
        y_ticks.append(sqrd_loading)

    x_ticks_sqLoad_org = x_ticks
    y_ticks_org = np.array(y_ticks)

    res = pd.DataFrame(columns=["x","y"])
    res['x'] = x_ticks
    res['y'] = y_ticks


    plt.figure()
    plt.plot(x_ticks, np.array(y_ticks)) 
    plt.xlabel('Variable Names') 
    plt.ylabel('Squared Loadings') 
    plt.title('Significance of Variables - Square Loadings (Org)') 
    plt.xticks(rotation=45)
    plt.savefig('./static/image/elbow/square_loading_org.png') 

    return json.dumps(res.to_json()) 



# Square Loadings Random
@app.route("/get_top_square_loadings_random", methods = ['POST', 'GET'])
def get_top_square_loadings_random(): 
    global x_ticks_sqLoad_random
    global y_ticks_random
    global squared_loadings_random

    files = glob.glob('./static/image/elbow/square_loading_random.png')
    for f in files:
        os.chmod(f, 0o777)
        if 'square_loading_random' in f:
            os.remove(f)

    num_PCA = int(request.form['num_PCA'])
    print("Num PCA - ",num_PCA)
    squared_loadings_random = {}
    for variable_id in range(0, len(eigen_vectors_random)):
        loadings = 0
        for PCA_Comp_Num in range(0, num_PCA):
            loadings = loadings + eigen_vectors_random[PCA_Comp_Num][variable_id] * eigen_vectors_random[PCA_Comp_Num][variable_id]
        squared_loadings_random[numerical_features[variable_id]] = loadings

    squared_loadings_random = sorted(squared_loadings_random.items(), key=lambda x: x[1], reverse=True)  

    x_ticks = []
    y_ticks = []
    for var_name, sqrd_loading in squared_loadings_random:
        x_ticks.append(var_name)
        y_ticks.append(sqrd_loading)

    x_ticks_sqLoad_random = x_ticks
    y_ticks_random = y_ticks

    res = pd.DataFrame(columns=["x","y"])
    res['x'] = x_ticks
    res['y'] = y_ticks

    plt.figure()
    plt.plot(x_ticks, np.array(y_ticks)) 
    plt.xlabel('Variable Names') 
    plt.ylabel('Squared Loadings') 
    plt.title('Significance of Variables - Square Loadings (Random)') 
    plt.xticks(rotation=45)
    plt.savefig('./static/image/elbow/square_loading_random.png') 

    return json.dumps(res.to_json()) 





# Square Loadings Stratified
@app.route("/get_top_square_loadings_stratified", methods = ['POST', 'GET'])
def get_top_square_loadings_stratified(): 
    global x_ticks_sqLoad_strat
    global y_ticks_stratified
    global squared_loadings_strat

    files = glob.glob('./static/image/elbow/square_loading_stratified.png')
    for f in files:
        os.chmod(f, 0o777)
        if 'square_loading_stratified' in f:
            os.remove(f)

    num_PCA = int(request.form['num_PCA'])
    print("Num PCA - ",num_PCA)
    squared_loadings_strat = {}
    for variable_id in range(0, len(eigen_vectors_stratified)):
        loadings = 0
        for PCA_Comp_Num in range(0, num_PCA):
            loadings = loadings + eigen_vectors_stratified[PCA_Comp_Num][variable_id] * eigen_vectors_stratified[PCA_Comp_Num][variable_id]
        squared_loadings_strat[numerical_features[variable_id]] = loadings

    squared_loadings_strat = sorted(squared_loadings_strat.items(), key=lambda x: x[1], reverse=True)  

    x_ticks = []
    y_ticks = []
    for var_name, sqrd_loading in squared_loadings_strat:
        x_ticks.append(var_name)
        y_ticks.append(sqrd_loading)

    x_ticks_sqLoad_strat = x_ticks
    y_ticks_stratified = y_ticks

    res = pd.DataFrame(columns=["x","y"])
    res['x'] = x_ticks
    res['y'] = y_ticks

    plt.figure()
    plt.plot(x_ticks, np.array(y_ticks)) 
    plt.xlabel('Variable Names') 
    plt.ylabel('Squared Loadings') 
    plt.title('Significance of Variables - Square Loadings (Stratified)') 
    plt.xticks(rotation=45)
    plt.savefig('./static/image/elbow/square_loading_stratified.png') 

    return json.dumps(res.to_json()) 




@app.route("/get_intrinsic_dimensionality_comparison", methods = ['POST', 'GET'])
def get_intrinsic_dimensionality_comparison(): 
    # files = glob.glob('./static/image/elbow/*')
    # for f in files:
    #     os.chmod(f, 0o777)
    #     if 'elbow_eigen_comparison' in f:
    #         os.remove(f)
    #     if 'elbow_eigen_comparison_bar' in f:
    #         os.remove(f)

    K = []
    for num in range(1,len(eigen_values_stratified) + 1):
        K.append(num)

    res = pd.DataFrame(columns=["x","y_org","y_random","y_strat"])
    res["x"] = K
    res["y_org"] = np.array(eigen_values_org[::-1])
    res["y_random"] = np.array(eigen_values_random[::-1])
    res["y_strat"] = np.array(eigen_values_stratified[::-1])

    plt.figure()
    plt.plot(K, np.array(eigen_values_org[::-1]), label="Original Data") 
    plt.plot(K, np.array(eigen_values_random[::-1]), label="Random Sample") 
    plt.plot(K, np.array(eigen_values_stratified[::-1]), label="Stratified Sample") 
    plt.legend(loc='upper right')
    plt.xlabel('Principal Component') 
    plt.ylabel('Eigen Values') 
    plt.title('The Scree Plot - Eigen Values vs Principal Components (Comparison)') 
    plt.savefig('./static/image/elbow/elbow_eigen_comparison.png') 


    eigen_values_ = eigen_values_org[::-1]
    sum_eigen = np.sum(eigen_values_)
    variance_org = []
    for index in range(0,len(eigen_values_)):
        variance_org.append(float(eigen_values_[index]/sum_eigen*100))

    cumulative_variance_org = np.cumsum(variance_org)

    eigen_values_ = eigen_values_random[::-1]
    sum_eigen = np.sum(eigen_values_)
    variance_random = []
    for index in range(0,len(eigen_values_)):
        variance_random.append(float(eigen_values_[index]/sum_eigen*100))

    cumulative_variance_random = np.cumsum(variance_random)

    eigen_values_ = eigen_values_stratified[::-1]
    sum_eigen = np.sum(eigen_values_)
    variance_strat = []
    for index in range(0,len(eigen_values_)):
        variance_strat.append(float(eigen_values_[index]/sum_eigen*100))

    cumulative_variance_strat = np.cumsum(variance_strat)

    res["y_cum_org"] = cumulative_variance_org
    res["y_cum_random"] = cumulative_variance_random
    res["y_cum_strat"] = cumulative_variance_strat
    res["y_var_org"] = variance_org
    res["y_var_random"] = variance_random
    res["y_var_strat"] = variance_strat

    plt.figure()
    plt.plot(K, cumulative_variance_org, label="Original Data") 
    plt.plot(K, cumulative_variance_random, label="Random Sample") 
    plt.plot(K, cumulative_variance_strat, label="Stratified Sample") 
    plt.bar(K, variance_org, label="Original Data") 
    plt.bar(K, variance_random, label="Random Sample") 
    plt.bar(K, variance_strat, label="Stratified Sample") 
    plt.legend(loc='upper right')
    plt.xlabel('Principal Component') 
    plt.ylabel('Variance Explained (%)') 
    plt.title('The Scree Plot - Eigen Values vs Principal Components (Comparison)') 
    plt.savefig('./static/image/elbow/elbow_eigen_comparison_bar.png') 


    # Mean bias
    org_mean = np.mean(np.array(eigen_values_org[::-1]))
    rand_mean = np.mean(np.array(eigen_values_random[::-1]))
    strat_mean = np.mean(np.array(eigen_values_stratified[::-1]))

    org_std = np.std(np.array(eigen_values_org[::-1]))
    rand_std = np.std(np.array(eigen_values_random[::-1]))
    strat_std = np.std(np.array(eigen_values_stratified[::-1]))

    rand_mean_bias = (rand_mean - org_mean) * 100 / org_mean
    strat_mean_bias = (strat_mean - org_mean) * 100 / org_mean

    rand_std_bias = (rand_std - org_std) * 100 / org_std
    strat_std_bias  = (strat_std - org_std) * 100 / org_std

    res["rand_mean_bias"] = rand_mean_bias
    res["strat_mean_bias"] = strat_mean_bias
    res["rand_std_bias"] = rand_std_bias
    res["strat_std_bias"] = strat_std_bias

    return json.dumps(res.to_json()) 



# Task 3
@app.route("/get_four_top_2PCA_org", methods = ['POST', 'GET'])
def get_four_top_2PCA_org(): 
    files = glob.glob('./static/image/elbow/*')
    for f in files:
        os.chmod(f, 0o777)
        if 'task3_1_org' in f:
            os.remove(f)
        
    stdScaler = StandardScaler()
    csvdata =  pd.read_csv('./static/dataset/avocado_dataset.csv', low_memory = False) 
    csvdata = csvdata.filter(numerical_features, axis=1)
    csvdata = stdScaler.fit_transform(csvdata)
   
    data = PCA(n_components=2)
    x_comp = csvdata
    data.fit(x_comp)
    x_comp = data.transform(x_comp)

    result_df = pd.DataFrame(x_comp, columns=["First_Component","Second_Component"])
    print("2d: ",result_df)

    res = pd.DataFrame(columns=["x","y"])
    res['x'] = result_df["First_Component"]
    res['y'] = result_df["Second_Component"]

    plt.figure()
    plt.scatter(result_df["First_Component"], result_df["Second_Component"])
    plt.xlabel('First Component') 
    plt.ylabel('Second Component') 
    plt.title('2 PCA (Original Data)') 
    plt.axis('equal')
    plt.savefig('./static/image/elbow/task3_1_org.png') 

    return json.dumps(res.to_json()) 

@app.route("/get_four_top_2PCA_random", methods = ['POST', 'GET'])
def get_four_top_2PCA_random(): 
    files = glob.glob('./static/image/elbow/*')
    for f in files:
        os.chmod(f, 0o777)
        if 'task3_1_random' in f:
            os.remove(f)
        
    data = PCA(n_components=2)
    stdScaler = StandardScaler()
    x_comp = stdScaler.fit_transform(random_sample)
    data.fit(x_comp)
    x_comp = data.transform(x_comp)

    result_df = pd.DataFrame(x_comp, columns=["First_Component","Second_Component"])
    print("2d random: ",result_df)
    
    res = pd.DataFrame(columns=["x","y"])
    res['x'] = result_df["First_Component"]
    res['y'] = result_df["Second_Component"]

    plt.figure()
    plt.scatter(result_df["First_Component"], result_df["Second_Component"])
    plt.xlabel('First Component') 
    plt.ylabel('Second Component') 
    plt.title('2 PCA (Random Sampling)') 
    plt.axis('equal')
    plt.savefig('./static/image/elbow/task3_1_random.png') 

    return json.dumps(res.to_json()) 



@app.route("/get_four_top_2PCA_strat", methods = ['POST', 'GET'])
def get_four_top_2PCA_strat(): 
       
    stdScaler = StandardScaler()
    cids = clustered_sample["clusterId"].values
    data = stdScaler.fit_transform(clustered_sample)

    pca = PCA(n_components=2)  
    projected = pd.DataFrame(pca.fit_transform(data),columns=["First","Second"])
    projected["clusterId"] = cids
    
    res = pd.DataFrame()
    for c_id in range(0,cids.max()+1):
        df = projected[projected["clusterId"] == c_id]
        res["x_"+str(c_id)] = df["First"].values
        res["y_"+str(c_id)] = df["Second"].values

    return json.dumps(res.to_json()) 


@app.route("/get_mds_org", methods = ['POST', 'GET'])
def get_mds_org():
    print("MDS For Original Data..")
    files = glob.glob('./static/image/elbow/*')
    for f in files:
        os.chmod(f, 0o777)
        if 'task3_2_org_euc' in f:
            os.remove(f)
        if 'task3_2_org_corr' in f:
            os.remove(f)
     
    stdScaler = StandardScaler()
    csvdata =  pd.read_csv('./static/dataset/avocado_dataset.csv', low_memory = False) 
    csvdata = csvdata.filter(numerical_features, axis=1)
    csvdata = stdScaler.fit_transform(csvdata)
  
    col = []
    MDS_Data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(csvdata, metric='euclidean')
    X = MDS_Data.fit_transform(similarity)
    col = pd.DataFrame(X, columns=["First_Component","Second_Component"])

    plt.figure()
    plt.scatter(col["First_Component"], col["Second_Component"])
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.axis('equal')
    plt.title('MDS via Euclidean Distance on Original Data')
    plt.savefig('./static/image/elbow/task3_2_org_euc.png') 

    res = pd.DataFrame()
    res['x_euc'] = col["First_Component"]
    res['y_euc'] = col["Second_Component"]

    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(csvdata, metric='correlation')
    X = mds_data.fit_transform(similarity)
    data_col = pd.DataFrame(X, columns=["First_Component","Second_Component"])

    plt.figure()
    plt.scatter(data_col["First_Component"], data_col["Second_Component"])
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.axis('equal')
    plt.title('MDS via Correlation Distance on Original Data')
    plt.savefig('./static/image/elbow/task3_2_org_corr.png') 

    res['x_corr'] = data_col["First_Component"]
    res['y_corr'] = data_col["Second_Component"]

    print('corr',res)
    return json.dumps(res.to_json()) 


@app.route("/get_mds_random", methods = ['POST', 'GET'])
def get_mds_random():
    print("MDS For Random Data..")
    files = glob.glob('./static/image/elbow/*')
    for f in files:
        os.chmod(f, 0o777)
        if 'task3_2_random_euc' in f:
            os.remove(f)
        if 'task3_2_random_corr' in f:
            os.remove(f)
     
    stdScaler = StandardScaler()
    randomSample = stdScaler.fit_transform(random_sample)
    MDS_Data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(randomSample, metric='euclidean')
    X = MDS_Data.fit_transform(similarity)
    col = pd.DataFrame(X, columns=["First_Component","Second_Component"])

   
    plt.figure()
    plt.scatter(col["First_Component"], col["Second_Component"])
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.axis('equal')
    plt.title('MDS via Euclidean Distance on Random Sample')
    plt.savefig('./static/image/elbow/task3_2_random_euc.png') 

    res = pd.DataFrame()
    res['x_euc'] = col["First_Component"]
    res['y_euc'] = col["Second_Component"]

    randomSample = stdScaler.fit_transform(random_sample)
    mds_data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(randomSample, metric='correlation')
    X = mds_data.fit_transform(similarity)
    data_col = pd.DataFrame(X, columns=["First_Component","Second_Component"])

    plt.figure()
    plt.scatter(data_col["First_Component"], data_col["Second_Component"])
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.axis('equal')
    plt.title('MDS via Correlation Distance on Random Data')
    plt.savefig('./static/image/elbow/task3_2_random_corr.png') 

    res['x_corr'] = data_col["First_Component"]
    res['y_corr'] = data_col["Second_Component"]

    return json.dumps(res.to_json()) 



@app.route("/get_mds_stratified", methods = ['POST', 'GET'])
def get_mds_stratified():
    print("MDS For Stratified Data..")
    files = glob.glob('./static/image/elbow/*')
    for f in files:
        os.chmod(f, 0o777)
        if 'task3_2_strat_euc' in f:
            os.remove(f)
        if 'task3_2_strat_corr' in f:
            os.remove(f)

    temp_df = clustered_sample.copy()
    cids = clustered_sample["clusterId"].values
    imp_cols = []
    imp_cols.append(squared_loadings_strat[0][0])
    imp_cols.append(squared_loadings_strat[1][0])
    print("unique cid : ",np.unique(cids), "  max : ",cids.max())

    stdScaler = StandardScaler()

    stratifiedSample = clustered_sample.filter(imp_cols, axis=1)
    stratifiedSample = pd.DataFrame(stdScaler.fit_transform(stratifiedSample))
    # stratifiedSample["clusterId"] =  cids

    print("strat sample : \n\n", stratifiedSample, "\n\n Shape: ",stratifiedSample.shape)
    
    # skip above
    stratifiedSample = stdScaler.fit_transform(clustered_sample)
    col = []
    MDS_Data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(stratifiedSample, metric='euclidean')
    X = MDS_Data.fit_transform(similarity)
    col = pd.DataFrame(X, columns=["First_Component","Second_Component"])
    col["clusterId"] =  cids

    plt.figure()
    for c_id in range(0,cids.max()+1):
        df = col[col["clusterId"] == c_id]
        plt.scatter(df["First_Component"], df["Second_Component"], color=colors[c_id])

    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.axis('equal')
    plt.title('MDS via Euclidean Distance on Stratified Sample')
    plt.savefig('./static/image/elbow/task3_2_strat_euc.png') 


    res = pd.DataFrame()
    for c_id in range(0,cids.max()+1):
        df = col[col["clusterId"] == c_id]
        res["x_euc_"+str(c_id)] = df["First_Component"].values
        res["y_euc_"+str(c_id)] = df["Second_Component"].values

    col = []
    stratifiedSample = stdScaler.fit_transform(stratified_sample)
    MDS_Data = manifold.MDS(n_components=2, dissimilarity='precomputed')
    similarity = pairwise_distances(stratifiedSample, metric='correlation')
    X = MDS_Data.fit_transform(similarity)
    col = pd.DataFrame(X, columns=["First_Component","Second_Component"])
    col["clusterId"] =  cids

    plt.figure()
    for c_id in range(0,cids.max()+1):
        df = col[col["clusterId"] == c_id]
        plt.scatter(df["First_Component"], df["Second_Component"], color=colors[c_id])

    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.axis('equal')
    plt.title('MDS via Correlation Distance on Stratified Sample')
    plt.savefig('./static/image/elbow/task3_2_strat_corr.png') 

    for c_id in range(0,cids.max()+1):
        df = col[col["clusterId"] == c_id]
        res["x_corr_"+str(c_id)] = df["First_Component"].values
        res["y_corr_"+str(c_id)] = df["Second_Component"].values

    print("strat : ",res)
    return json.dumps(res.to_json()) 





# Task 3 part 3
@app.route("/get_four_top3PCA_org", methods = ['POST', 'GET'])
def get_four_top3PCA_org():
    print("Top 3 PCA For Original Data..")
    # files = glob.glob('./static/image/elbow/*')
    # for f in files:
    #     os.chmod(f, 0o777)
    #     if 'task3_3_org' in f:
    #         os.remove(f)
       
    top_3PCA = []
    for i in range(0,3):
        top_3PCA.append(squared_loadings_org[i][0])
    
    stdScaler = StandardScaler()
    csvdata =  pd.read_csv('./static/dataset/avocado_dataset.csv', low_memory = False) 
    csvdata = csvdata.filter(numerical_features, axis=1)
    csvdata = pd.DataFrame(stdScaler.fit_transform(csvdata), columns=csvdata.columns)
   
    data = PCA(n_components=3)
    x_comp = csvdata
    data.fit(x_comp)
    x_comp = data.transform(x_comp)
    result_df = pd.DataFrame(x_comp, columns=top_3PCA)
    

    sns_plot = sns.pairplot(result_df)
    sns_plot.savefig('./static/image/elbow/task3_3_org.png') 

    return json.dumps(result_df.to_json()) 



@app.route("/get_four_top3PCA_random", methods = ['POST', 'GET'])
def get_four_top3PCA_random():
    print("Top 3 PCA For Random Sample..")
    files = glob.glob('./static/image/elbow/*')
    for f in files:
        os.chmod(f, 0o777)
        if 'task3_3_random' in f:
            os.remove(f)
       
    top_3PCA = []
    for i in range(0,3):
        top_3PCA.append(squared_loadings_random[i][0])

    data = PCA(n_components=3)
    stdScaler = StandardScaler()
    x_comp = stdScaler.fit_transform(random_sample)
    data.fit(x_comp)
    x_comp = data.transform(x_comp)
    
    randomSample = pd.DataFrame(x_comp,columns=top_3PCA)
   
    sns_plot = sns.pairplot(randomSample)
    sns_plot.savefig('./static/image/elbow/task3_3_random.png') 

    return json.dumps(randomSample.to_json()) 
    



@app.route("/get_four_top3PCA_stratified", methods = ['POST', 'GET'])
def get_four_top3PCA_stratified():
    print("Top 3 PCA For Stratified Sample..")
    files = glob.glob('./static/image/elbow/*')
    stdScaler = StandardScaler()

    for f in files:
        os.chmod(f, 0o777)
        if 'task3_3_strat' in f:
            os.remove(f)
       
    top_3PCA = []
    cids = clustered_sample["clusterId"].values
    for i in range(0,3):
        top_3PCA.append(squared_loadings_strat[i][0])

    stratifiedSample = pd.DataFrame(stdScaler.fit_transform(clustered_sample))
    
    pca = PCA(n_components=3) 
    projected = pd.DataFrame(pca.fit_transform(stratifiedSample),columns=top_3PCA)
    projected["clusterId"] = cids
    
    sns_plot = sns.pairplot(projected, hue="clusterId")
    sns_plot.savefig('./static/image/elbow/task3_3_strat.png') 


    return json.dumps(projected.to_json()) 





if __name__ == "__main__":
    app.run(debug=True)