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
    
# For Distributed/Federated VAMP 

MSE_matrix_final=[]
monte=50

MSE_matrix=[]
for iteration in range (0,monte):
    print(iteration)
    N=200
    colours=['b','g','r','c']
    num_agents=10
    count=0
    L=np.random.uniform(0,1,size=(M,M))
    m=np.zeros((N,1))
    gamma_x=1
    x_02= np.random.normal(0,1,size=(N,1))
    x = x_02

    
    MSE_1_correlation=[]
    for correlation in range (num_agents):
        
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
        W=[]
        COV=[]
        if correlation>0:

            for i in range (correlation):

                size_M=int((N/2)//num_agents)
                M.append(size_M)
                A.append(np.random.normal(0,1,size=(M[i], N)))
                w,cov=correlated_noise(L,A[i].shape[0])
                W.append(w)
                COV.append(cov)
                y.append(np.matmul(A[i],x) + W[i])
            
        for l in range (correlation,num_agents):
                
            size_M=int((N/2)//num_agents)
            M.append(size_M)
            A.append(np.random.normal(0,1,size=(M[l], N)))
            noise_var=(np.mean((np.matmul(A[l],x))**2)*10**(-20/10))
            gamma_w.append(noise_var**-2)
            y.append(np.matmul(A[l],x) + np.random.normal(0.0, noise_var, (M[l],1)))

            

        for counts in range (num_agents):
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
                
                if j < correlation:

                    x_hat_2[j]= np.matmul((np.linalg.inv(np.matmul(np.matmul(np.transpose(A[j]),COV[j]),A[j]) + gamma_2[j]*np.eye(N))),
                           (np.matmul(np.matmul(np.transpose(A[j]),COV[j]),y[j]) + gamma_2[j]*r_2[j]))  
            
                    alpha_2[j]= gamma_2[j]*np.trace(np.linalg.inv(np.matmul(np.matmul(np.transpose(A[j]),COV[j]),A[j]) + gamma_2[j]*np.eye(N)))/N
                    
                else:
                    #print(j,correlation)
                    x_hat_2[j]= (np.matmul((np.linalg.inv(gamma_w[j-correlation]*np.matmul(np.transpose(A[j]),A[j]) + gamma_2[j]*np.eye(N))),(gamma_w[j-correlation]*np.matmul(np.transpose(A[j]),y[j]) + gamma_2[j]*r_2[j])))
                    alpha_2[j]= gamma_2[j]*np.trace(np.linalg.inv(gamma_w[j-correlation]*np.matmul(np.transpose(A[j]),A[j]) + gamma_2[j]*np.eye(N)))/N
                eta_2[j]= gamma_2[j]/alpha_2[j]
                gamma_1[j]= eta_2[j] - gamma_2[j]

                r_1[j]= (eta_2[j]*x_hat_2[j] - gamma_2[j]*r_2[j])/gamma_1[j]
                r_1[j]= ((1-reg)*old_r_1[j] + reg*r_1[j])
                old_r_1[j]=r_1[j]

                gamma_1[j]= ((1-reg)*old_gamma_1[j] + reg*gamma_1[j])
                old_gamma_1[j]=gamma_1[j]        
            
                MSE_f =MSE_f+(np.mean((np.matmul(A[j],x_02) - np.matmul(A[j],x_hat_2[j]))**2)/np.mean(np.matmul(A[j],x_02)**2))
            MSE=MSE_f/num_agents

            #print("mse ", MSE)
        
        MSE_1_correlation.append(MSE) # [ 14 values]
    MSE_matrix.append(MSE_1_correlation) 

   
final_matrix=[]

sums=[] 
lists=[]
for j in MSE_matrix:
    sums.append(j)
#import pdb; pdb.set_trace()
result_list = add_n_lists(sums)
result_list=divide_list_by_scalar(monte,result_list)
final_matrix.append(result_list) 


# For Standard VAMP 

N=200
M=100
# start = 0
# end = 20
# step = 0.2
# SNR = []

# value = start
# while value <= end:
#     SNR.append(value)
#     value += step
#SNR = list(range(0, 20, 1))
SNR=[20]
MSE_final=[]

#SSNR= 20
rho=1
reg=0.7
m1=0
gamma_1=1

#mean and precision of prior 
m=np.zeros((N,1))
gamma_x=1

x_02= np.random.normal(0,1,size=(N,1))
#mask= np.random.binomial(1, rho, (N,1)) 
#x = mask * x_02
x=x_02

for i in SNR:


    #import pdb; pdb.set_trace()
    
    A=np.random.normal(0,1,(M,N))
    noise_var= (np.mean((np.matmul(A,x))**2))*10**(-i/10)
    gamma_w=noise_var**-2

    #y = np.matmul(A,x) + np.random.normal(0.0, noise_var, (M,1))
    
    w,cov=correlated_noise(A.shape[0])
    y1=np.matmul(A,x)
    y= y1 + w
    
    snr= 10*np.log10((np.linalg.norm(y1)**2)/np.linalg.norm(w)**2)



    #x_01= np.random.normal(m, sigma, (N,1))


    #Denoisinh
    x_hat_1 = np.random.normal(0.0, 1.0, (N,1))
    alpha_1 = 1.0
    eta_1 = 1.0
    gamma_2 = 1.0
    r_2 = np.random.normal(0.0, 1.0, (N,1))


    #LMMSE 
    x_hat_2 = np.random.normal(0.0, 1.0, (N,1))
    alpha_2 = 1.0
    eta_2 = 1.0
    gamma_1 = 1.0
    r_1 = np.random.normal(0.0, 1.0, (N,1))

    #old valuesexit()

    old_r_1= np.zeros((N,1))
    old_r_2= np.zeros((N,1))
    old_gamma_1=1
    old_gamma_2=1


    for i in range (10):
        #Denoising Updation 
        #print(i,gamma_1)

        gamma_post= (gamma_x+gamma_1)
        x_hat_1= rho*(m*gamma_x + r_1*gamma_1)*(gamma_post)**-1
        alpha_1= rho*gamma_1*(gamma_post)**-1
        eta_1= gamma_1/alpha_1
        

        
        gamma_2= eta_1 - gamma_1 
        r_2= (eta_1*x_hat_1 - gamma_1*r_1)/gamma_2
        #import pdb; pdb.set_trace()
        #r_2= ((1-reg)*old_r_2*old_gamma_2 + reg*r_2*gamma_2)*(1/((1-reg)*old_gamma_2 + reg*gamma_2))
        r_2= ((1-reg)*old_r_2 + reg*r_2)
        old_r_2= r_2
        gamma_2=(1-reg)*old_gamma_2 + reg*gamma_2
        old_gamma_2= gamma_2
        

        #LMMSE updation

        x_hat_2= np.matmul((np.linalg.inv(np.matmul(np.matmul(np.transpose(A),cov),A) + gamma_2*np.eye(N))),
                           (np.matmul(np.matmul(np.transpose(A),cov),y) + gamma_2*r_2))  
        #print(x_hat_2,x_02)
        alpha_2= gamma_2*np.trace(np.linalg.inv(np.matmul(np.matmul(np.transpose(A),cov),A) + gamma_2*np.eye(N)))/N
        eta_2= gamma_2/alpha_2
        #import pdb; pdb.set_trace()
        gamma_1= eta_2 - gamma_2
        r_1= (eta_2*x_hat_2 - gamma_2*r_2)/gamma_1
    # r_1= ((1-reg)*old_r_1*old_gamma_1 + reg*r_1*gamma_1)*(1/((1-reg)*old_gamma_1 + reg*gamma_1))
        r_1= ((1-reg)*old_r_1 + reg*r_1)
        old_r_1=r_1
        gamma_1=(1-reg)*old_gamma_1 + reg*gamma_1
        old_gamma_1=gamma_1
        MSE = (np.mean((np.matmul(A,x_02) - np.matmul(A,x_hat_2))**2)/np.mean(np.matmul(A,x_02)**2))
        print("MSE",MSE)

MSE=MSE*np.ones(num_agents)

x_03=np.linspace(1,num_agents,num_agents)
plt.title('MSE vs Num of correlated agents (gaussian-20dB)')
plt.semilogy(x_03,result_list)
plt.semilogy(x_03,MSE)

plt.xlabel('number of correlated agents')
plt.ylabel('MSE')
plt.legend(['Federated VAMP','Standard VAMP'])
plt.savefig('correlation3.png')
plt.show()
