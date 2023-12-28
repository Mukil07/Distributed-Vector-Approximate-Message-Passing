import time

start = time.time()
import math
import numpy as np
import matplotlib.pyplot as plt
#np.random.seed(13)


def correlated_noise(L,M):
    #n=np.random.normal(0,1,size=(M,M))

    #L=np.random.normal(0,1,size=(M,M))
    covariance=np.matmul(L,L.T)
    covariance=covariance/((np.linalg.norm(covariance))**0.41)
    mean= np.zeros(M)
    return np.random.multivariate_normal(mean, covariance, size=1),covariance

def add_n_lists(lists):
    if len(lists) < 2:
        raise ValueError("At least two lists are required for addition.")
    min_length = min(len(lst) for lst in lists)
    zipped_lists = zip(*lists)
    result = [sum(elements) for elements in zipped_lists]

    return result

def divide_list_by_scalar(scalar, input_list):
    return [element/scalar for element in input_list]

MSE_matrix_final=[]
monte=2

MSE_matrix=[]
for iteration in range (0,monte):
    SNR_for_each_agent=[1,2,3,4,5,6,7,8,9,10]
    print(iteration)
    N=200
    colours=['b','g','r','c']
    num_agents=16
    count=0
    #L=np.random.uniform(0,1,size=(M,M))
    m=np.zeros((N,1))
    gamma_x=1
    x_02= np.random.normal(0,1,size=(N,1))
    x = x_02
    MSE_1_correlation=[]
    for i in SNR_for_each_agent:

        var= np.random.normal(15,i,size=num_agents)
        
        

        rho=1
        reg=1

        A=[]
        gamma_w=[]
        M=[]
        y=[]

        #Denoising

        x_hat_1 = [np.random.normal(0.0, 1.0, (N,1))]
        alpha_1 = []
        eta_1 = []
        gamma_2=[]
        r_2=[]

        #LMMSE
        x_hat_2 = []
        alpha_2 = []
        eta_2 = []
        gamma_1 = []
        r_1 = []

        #for momentum - regularization
        old_r_2=[]
        old_r_1=[]
        old_gamma_2=[]
        old_gamma_1=[]


        for l in range (num_agents):
            
            size_M=int((N/2)//num_agents)
            M.append(size_M)
            A.append(np.random.normal(0,1,size=(M[l], N)))
            noise_var=(np.mean((np.matmul(A[l],x))**2)*10**(-var[l]/10))
            gamma_w.append(noise_var**-2)
            y.append(np.matmul(A[l],x) + np.random.normal(0.0, noise_var, (M[l],1)))

            #denoising
            gamma_2.append(1)
            r_2.append(np.random.normal(0.0, 1.0, (N,1)))

            #LMMSE
            x_hat_2.append(np.random.normal(0.0, 1.0, (N,1)))
            alpha_2.append(1)
            eta_2.append(1)
            gamma_1.append(1)
            r_1.append(np.random.normal(0.0, 1.0,(N,1)))

            old_r_2.append(np.zeros((N,1)))
            old_r_1.append(np.zeros((N,1)))
            old_gamma_2.append(1)
            old_gamma_1.append(1)


            r_new=[]

            gamma_temp=[]
            r_temp=[]
            gamma_x=1
            gamma_post=[]
            MSE_final=[]
            num_iterations=25
        for i in range (num_iterations):

            #Denoising Updation

            temp_g=0
            temp_r=0
            for k in range(num_agents):
                temp_g=temp_g+gamma_1[k]
                temp_r=temp_r+r_1[k]*gamma_1[k]
            gamma_temp.append((temp_g))
            r_temp.append((gamma_temp[i]**-1)*(temp_r))

            gamma_post.append((gamma_x+gamma_temp[i]))
            x_hat_1.append(rho*(gamma_post[i]**-1)*(m*gamma_x + r_temp[i]*gamma_temp[i]))
            alpha_1.append(rho*(gamma_post[i]**-1)*gamma_temp[i])
            eta_1.append(gamma_temp[i]/alpha_1[i])

            MSE_f=0
            #LMMSE updation
            for j in range (num_agents):

                gamma_2[j]= (eta_1[i] - gamma_1[j])
                r_2[j]= ((eta_1[i]*x_hat_1[i+1] - gamma_1[j]*r_1[j])/gamma_2[j])

                r_2[j]= ((1-reg)*old_r_2[j] + reg*r_2[j])
                old_r_2[j]=r_2[j]

                gamma_2[j]= ((1-reg)*old_gamma_2[j] + reg*gamma_2[j])
                old_gamma_2[j]=gamma_2[j]


                x_hat_2[j]= (np.matmul((np.linalg.inv(gamma_w[j]*np.matmul(np.transpose(A[j]),A[j]) + gamma_2[j]*np.eye(N))),(gamma_w[j]*np.matmul(np.transpose(A[j]),y[j]) + gamma_2[j]*r_2[j])))
                alpha_2[j]= gamma_2[j]*np.trace(np.linalg.inv(gamma_w[j]*np.matmul(np.transpose(A[j]),A[j]) + gamma_2[j]*np.eye(N)))/N
                eta_2[j]= gamma_2[j]/alpha_2[j]
                gamma_1[j]= eta_2[j] - gamma_2[j]

                r_1[j]= (eta_2[j]*x_hat_2[j] - gamma_2[j]*r_2[j])/gamma_1[j]
                r_1[j]= ((1-reg)*old_r_1[j] + reg*r_1[j])
                old_r_1[j]=r_1[j]

                gamma_1[j]= ((1-reg)*old_gamma_1[j] + reg*gamma_1[j])
                old_gamma_1[j]=gamma_1[j]

                MSE_f =MSE_f+(np.mean((np.matmul(A[j],x_02) - np.matmul(A[j],x_hat_2[j]))**2)/np.mean(np.matmul(A[j],x_02)**2))
        MSE=MSE_f/num_agents
        MSE_1_correlation.append(MSE) # [ 10 values}
        
    MSE_matrix.append(MSE_1_correlation) #[ monte carlo iterations]
    #import pdb; pdb.set_trace()


sums=[]
lists=[]
for j in MSE_matrix:
    sums.append(j)
#import pdb; pdb.set_trace()
result_list1 = add_n_lists(sums)
result_list1=divide_list_by_scalar(monte,result_list1)



################################################################################################################################################


x_03=np.linspace(1,10,10)
plt.title('NRMSE vs Num of heterogenous agents (gaussian)')
#import pdb; pdb.set_trace()
plt.semilogy(SNR_for_each_agent,result_list1)
# plt.semilogy(x_03,result_list2)
# plt.semilogy(x_03,result_list3)
# plt.semilogy(x_03,result_list4)

plt.xlabel('Variance of noise')
plt.ylabel('NRMSE')
plt.legend(['Federated VAMP(heterogenous)'])
plt.savefig('mukil.jpg')
plt.show()
