import time

start = time.time()
import math
import numpy as np
import matplotlib.pyplot as plt


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
monte=100
for iteration in range (1,monte):
    print(iteration)
    MSE_matrix=[]
    #N=int(input("enter the number of unknowns(N):  "))
    N=300
    SNR=[15]
    colours=['b','g','r','c','m']

    #num_agents=list(range(1, 11, 1))
    num_agents=[1,2,4,8,16]
    #num_agents=[50]
    count=0

    for snrs in SNR:
        MSE_final=[]

        for agents in num_agents:


            rho=1
            reg=1

            A=[]
            gamma_w=[]
            M=[]
            y=[]

            #mean and precision of prior 
            m=np.zeros((N,1))
            gamma_x=1
            x_02 = np.random.choice([-1, 1], size=(N,1))
            #x_02= np.random.normal(0,1,size=(N,1))
            #mask=np.random.binomial(1.0, rho, (N,1))
            x = x_02


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

            #n= int(input("enter the number of agents : "))
            n=agents
            print(n)
            for i in range (n):
                
                #size_M= int(input("enter the size of A_%d : "%i))
                size_M=int((N/2)//n)
                #snr=int(input("enter the SNR_%d : "%i))
                #SNR.append(snr)
                M.append(size_M)
                A.append(np.random.normal(0,1,size=(M[i], N)))
                noise_var=(np.mean((np.matmul(A[i],x))**2)*10**(-snrs/10))
                gamma_w.append(noise_var**-2)
                y.append(np.matmul(A[i],x) + np.random.normal(0.0, noise_var, (M[i],1)))

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
            num_iterations=50
            for i in range (num_iterations):

                #Denoising Updation 
                #old_x_hat_1= x_hat_1[i]
                temp_g=0
                temp_r=0
                for k in range(n):
                    temp_g=temp_g+gamma_1[k]
                    temp_r=temp_r+r_1[k]*gamma_1[k]
                gamma_temp.append((temp_g))
                r_temp.append((gamma_temp[i]**-1)*(temp_r))

                vectorized_tanh = np.vectorize(math.tanh)
                x_hat_1.append(rho*vectorized_tanh(r_temp[i]*gamma_temp[i]))
                alpha_12= rho*(np.mean((1-(vectorized_tanh(r_temp[i]*gamma_temp[i])**2))))*gamma_temp[i]
                alpha_1.append(np.maximum(alpha_12,1e-10))
                eta_1.append(gamma_temp[i]/alpha_1[i])

                MSE_f=0
                #LMMSE updation
                for j in range (n):

                    gamma_2[j]= (eta_1[i] - gamma_1[j]) 
                    r_2[j]= ((eta_1[i]*x_hat_1[i+1] - gamma_1[j]*r_1[j])/gamma_2[j])

                    r_2[j]= ((1-reg)*old_r_2[j] + reg*r_2[j])
                    old_r_2[j]=r_2[j]

                    gamma_2[j]= ((1-reg)*old_gamma_2[j] + reg*gamma_2[j])
                    old_gamma_2[j]=gamma_2[j]
                    
                    x_hat_2[j]= (np.matmul((np.linalg.inv(gamma_w[j]*np.matmul(np.transpose(A[j]),A[j]) + gamma_2[j]*np.eye(N))),(gamma_w[j]*np.matmul(np.transpose(A[j]),y[j]) + gamma_2[j]*r_2[j])))
                    
                    alpha_2[j]= gamma_2[j]*np.trace(np.linalg.inv(gamma_w[j]*np.matmul(np.transpose(A[j]),A[j]) + gamma_2[j]*np.eye(N)))/N
                    eta_2[j]= gamma_2[j]/alpha_2[j]
                    #print(eta_2[j])

                    gamma_1[j]= eta_2[j] - gamma_2[j]
                    #print(i,j,x_hat_2[j])
                    r_1[j]= (eta_2[j]*x_hat_2[j] - gamma_2[j]*r_2[j])/gamma_1[j]

                    #print(gamma_1[j])
                    r_1[j]= ((1-reg)*old_r_1[j] + reg*r_1[j])
                    old_r_1[j]=r_1[j]

                    gamma_1[j]= ((1-reg)*old_gamma_1[j] + reg*gamma_1[j])
                    old_gamma_1[j]=gamma_1[j]        
                
                    MSE_f =MSE_f+(np.mean((np.matmul(A[j],x_02) - np.matmul(A[j],x_hat_2[j]))**2)/np.mean(np.matmul(A[j],x_02)**2))
                MSE=MSE_f/n

                #print("mse ", MSE)
            
                MSE_final.append(MSE) # [ 14 values]
            MSE_matrix.append(MSE_final) # has format -- [a1] [ 5 values ]
    MSE_matrix_final.append(MSE_matrix)    #[ 2 values ]


count=0   
   
final_matrix=[]
for i in range (len(num_agents)):
    sums=[] 
    lists=[]
    for j in MSE_matrix_final:
        sums.append(j[i])
    #import pdb; pdb.set_trace()
    result_list = add_n_lists(sums)
    result_list=divide_list_by_scalar(monte,result_list)
    final_matrix.append(result_list) 


for i in final_matrix:
    x_03=np.linspace(1,num_iterations,num_iterations)
    plt.title('MSE vs Num of iterations - Binary(10dB)')
    plt.semilogy(x_03,i)

plt.xlabel('number of iterations')
plt.ylabel('MSE')
plt.legend(['agents_1','agents_2','agents_4','agents_8','agents_16'])
plt.show()
    
