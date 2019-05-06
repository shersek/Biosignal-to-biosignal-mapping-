import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from statsmodels import robust
import pickle
import multi_stat_analysis
import scipy.stats as st

directory = '/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials'
fileObject = open(directory + '/Results and Figures/'+'model_test_results_pickle_different_axes','rb')
(list_model_names, list_models_pearson , list_models_i, list_models_j, list_models_k) = pickle.load(fileObject)

#boxplot for pearson correlation
list_frames = []
for u in range(len(list_model_names)):
    list_frames.append(pd.DataFrame(data={'Pearson':list_models_pearson[u] , 'Type': list_model_names[u][-20:-15] + list_model_names[u][-6::]}))
dfr_results = pd.concat(list_frames)

fig = plt.figure(figsize=(7,3));
sns.set_style('whitegrid', {'grid.linestyle':'--'})
ax=sns.boxplot(x='Type', y='Pearson' , data=dfr_results , linewidth=1.5  ,
               palette=sns.xkcd_palette(['cherry red' , 'cerulean blue', 'green' , 'orange', 'light purple' , 'yellow' , 'sky blue']) ,  showfliers=False )
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.ylim(-0.31,1)
plt.tight_layout()
plt.show()
plt.savefig("/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Results_Axes_PCC.svg")

fig = plt.figure(figsize=(3.5,3));
sns.set_style('whitegrid', {'grid.linestyle':'--'})
ax=sns.boxplot(x='Type', y='Pearson' , data=dfr_results , linewidth=1.5  ,
               palette=sns.xkcd_palette(['cherry red' , 'cerulean blue', 'green' , 'orange', 'light purple' , 'yellow' , 'sky blue']) ,  showfliers=False )
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.ylim(-0.31,1)
plt.tight_layout()
plt.show()
plt.savefig("/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Results_Axes_PCC_2.svg")

#boxplot for timing intervals
list_frames = []
for u in range(len(list_model_names)):
    list_frames.append(pd.DataFrame(data={'Error': [v for v in list_models_i[u] if v>=0] , 'Model': list_model_names[u][-20:-15] + list_model_names[u][-6::]  , 'Type': 'R-I' }  ))
    list_frames.append(pd.DataFrame(data={'Error': [v for v in list_models_j[u] if v>=0] , 'Model': list_model_names[u][-20:-15] + list_model_names[u][-6::]  , 'Type': 'R-J' }  ))
    list_frames.append(pd.DataFrame(data={'Error': [v for v in list_models_k[u] if v>=0] , 'Model': list_model_names[u][-20:-15] + list_model_names[u][-6::]  , 'Type': 'R-K' }  ))
dfr_results = pd.concat(list_frames)

fig = plt.figure(figsize=(7,3))
sns.set_style('whitegrid', {'grid.linestyle':'--'})
ax=sns.boxplot(x='Model', y='Error' , hue = 'Type' , data=dfr_results , linewidth=1.5  ,
               palette=sns.xkcd_palette(['cherry red' , 'cerulean blue', 'green']) ,  showfliers=False )
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.ylim(-2,52)
plt.tight_layout()
plt.show()
plt.savefig("/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Results_Axes_Timing.svg")

#boxplot for timing intervals
list_frames = []
for u in range(len(list_model_names)):
    list_frames.append(pd.DataFrame(data={'Error': [v for v in list_models_i[u] if v>=0] , 'Model': list_model_names[u][-20:-15] + list_model_names[u][-6::]  , 'Type': 'R-I' }  ))
    list_frames.append(pd.DataFrame(data={'Error': [v for v in list_models_j[u] if v>=0] , 'Model': list_model_names[u][-20:-15] + list_model_names[u][-6::]  , 'Type': 'R-J' }  ))
    list_frames.append(pd.DataFrame(data={'Error': [v for v in list_models_k[u] if v>=0] , 'Model': list_model_names[u][-20:-15] + list_model_names[u][-6::]  , 'Type': 'R-K' }  ))
dfr_results = pd.concat(list_frames)

fig = plt.figure(figsize=(3.5,3))
sns.set_style('whitegrid', {'grid.linestyle':'--'})
ax=sns.boxplot(x='Model', y='Error' , hue = 'Type' , data=dfr_results , linewidth=1.5  ,
               palette=sns.xkcd_palette(['cherry red' , 'cerulean blue', 'green']) ,  showfliers=False )
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(0.5)
ax.spines['right'].set_linewidth(0.5)
ax.spines['top'].set_linewidth(0.5)
ax.spines['bottom'].set_linewidth(0.5)
plt.ylim(-2,52)
plt.tight_layout()
plt.show()
plt.savefig("/media/sinan/9E82D1BB82D197DB/RESEARCH VLAB work on/Gyroscope SCG project/Deep Learning Paper Code and Materials/Results and Figures/Results_Axes_Timing_2.svg")



#print results table
for v,model in enumerate(list_model_names):
    #print results median +/- mad



    print(model +  '& {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) & {:.2f} ({:.2f}) \\\\'.format(
                                                                                    np.median(np.array(list_models_pearson[v])),
                                                                                    robust.mad(np.array(list_models_pearson[v])),
                                                                                    np.median(np.array([u for u in list_models_i[v] if u >= 0])),
                                                                                    robust.mad(np.array([u for u in list_models_i[v] if u >= 0])),
                                                                                    np.median(np.array([u for u in list_models_j[v] if u >= 0])),
                                                                                    robust.mad(np.array([u for u in list_models_j[v] if u >= 0])),
                                                                                    np.median(np.array([u for u in list_models_k[v] if u >= 0])),
                                                                                    robust.mad(np.array([u for u in list_models_k[v] if u >= 0])) ))


    print('\hline')

    # print(model+' & & & &  \\\\')
    # print('Median & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\'.format( np.median(np.array(list_models_pearson[v])) ,
    #                                                                 np.median(np.array([u for u in list_models_i[v] if u >= 0])),
    #                                                                 np.median(np.array([u for u in list_models_j[v] if u >= 0])) ,
    #                                                                 np.median(np.array([u for u in list_models_k[v] if u >= 0]))) )
    # print('MAD & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\'.format( robust.mad(np.array(list_models_pearson[v])) ,
    #                                                              robust.mad(np.array([u for u in list_models_i[v] if u >= 0])),
    #                                                              robust.mad(np.array([u for u in list_models_j[v] if u >= 0])) ,
    #                                                              robust.mad(np.array([u for u in list_models_k[v] if u >= 0]))) )
    # print('\hline')


def multi_comparison_testing(dfr_results):
    # perform Friedman test
    results_matrix = dfr_results.values
    degrees_of_freedom = results_matrix.shape[1] - 1
    number_of_segments = results_matrix.shape[0]
    chi_squared, p_val = st.friedmanchisquare(*(results_matrix[:, i] for i in range(results_matrix.shape[1])))
    print('RESULTS TRANSCRIPT')
    print(
        'There was a statistically significant difference in results depending on which model was chosen.')
    print('χ2(' + str(degrees_of_freedom), ')=', str(chi_squared), ', p=', str(p_val), ', n=', str(number_of_segments),
          '(Friedman Test).')
    print('For post hoc testing, performed Wilcoxon signed rank test to pairs of models to be compared.')
    print('Benjamini-Hochberg Correction was done on the p-values from the post hoc testing.')
    print('Figure shows statistically significant differences where Benjamini-Hochberg corrected p<0.05')

    # perform post hoc test: Wilcoxon signed rank testing with Benjamini-Hochberg adjustment
    list_groups = dfr_results.columns.tolist()
    index_1 = np.arange(len(list_groups))
    index_2 = np.arange(len(list_groups))
    significanceMatrix = np.zeros([len(list_groups), len(list_groups)])
    p_val_vector = []
    np.seterr(divide='ignore', invalid='ignore')

    # perform pariwise Wilcoxon signed rank testing
    index_p_val_vector = 0
    for u1 in index_1:
        for u2 in index_2:
            if u1 > u2:
                stat_test = st.wilcoxon(dfr_results[list_groups[u1]].values,
                                        dfr_results[list_groups[u2]].values)
                p_val_vector.append(stat_test.pvalue)
                index_p_val_vector = index_p_val_vector + 1

    # perform benjamini-hochberg adjustment
    reject_, pvals_corrected_, alphacSidak, alphacBonf = multi_stat_analysis.multipletests(np.array(p_val_vector),
                                                                                           alpha=0.05, method='fdr_bh',
                                                                                           is_sorted=False,
                                                                                           returnsorted=False)
    # unpack hypothesis rejection vector to a matrix
    index_p_val_vector = 0
    for u1 in index_1:
        for u2 in index_2:
            if u1 > u2:
                significanceMatrix[u1, u2] = (reject_[index_p_val_vector])  # True if significance detected
                index_p_val_vector = index_p_val_vector + 1
            else:
                significanceMatrix[u1, u2] = 0

    # plot the significance matrix from post hoc testing
    fig = plt.figure(figsize=(12, 9));
    sns.set_style('whitegrid', {'grid.linestyle': '--'})
    mask = np.zeros_like(significanceMatrix)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(significanceMatrix, mask=mask, xticklabels=list_groups, yticklabels=list_groups, linewidth=1,
                cmap='OrRd')
    plt.show()

#Pearson Correlation Results
list_frames = []
for u in range(len(list_model_names)):
    list_frames.append( pd.DataFrame(data={list_model_names[u][-20:-15] + list_model_names[u][-6::] : list_models_pearson[u]  }  ))
dfr_results_pearson = pd.concat(list_frames,axis=1)
multi_comparison_testing(dfr_results_pearson)

#R-I Results
list_frames = []
for u in range(len(list_model_names)):
    list_frames.append( pd.DataFrame(data={list_model_names[u][-20:-15] + list_model_names[u][-6::] : np.array([v if v >= 0 else np.nan for v in list_models_i[u] ]) }  ))
dfr_results_ri = pd.concat(list_frames,axis=1)
dfr_results_ri=dfr_results_ri.dropna(axis = 0, how='any')
multi_comparison_testing(dfr_results_ri)

#R-J Results
list_frames = []
for u in range(len(list_model_names)):
    list_frames.append( pd.DataFrame(data={list_model_names[u][-20:-15] + list_model_names[u][-6::] : np.array([v if v >= 0 else np.nan for v in list_models_j[u] ]) }  ))
dfr_results_rj = pd.concat(list_frames,axis=1)
dfr_results_rj=dfr_results_rj.dropna(axis = 0, how='any')
multi_comparison_testing(dfr_results_rj)

#R-K Results
list_frames = []
for u in range(len(list_model_names)):
    list_frames.append( pd.DataFrame(data={list_model_names[u][-20:-15] + list_model_names[u][-6::] : np.array([v if v >= 0 else np.nan for v in list_models_k[u] ]) }  ))
dfr_results_rk = pd.concat(list_frames,axis=1)
dfr_results_rk=dfr_results_rk.dropna(axis = 0, how='any')
multi_comparison_testing(dfr_results_rk)

