import numpy as np
import cvxpy as cp

from scipy.optimize import minimize

# Importing standard Qiskit libraries
from qiskit import QuantumCircuit, transpile, Aer, execute
from qiskit.circuit import Parameter
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library.standard_gates import RYGate, XXPlusYYGate

from src import QAOA_utilities as QAOAut

# -------------- Main Optimization Routines: ------------------

def optimize_links_fixedt(BW_max, rho, eta, alpha, beta, t, lambda_s, p_param = 3 ):
    # optimize network links using QAOA algorithm
    # BW_max: max bandwidth of network
    # rho: max power for any user
    # eta: [vector of dimension K] power control for each user
    # alpha, beta: [matrices M x K] network parameters (see notes)
    # t: current lower bound for sinr
    # p_param: depth of the QAOA circuit
    # lambda_s: [vector of dim K] set of penalization parameters for soft contraints (Lagrange multipliers)

    M, K = beta.shape[0], beta.shape[1]

    X_mat = np.zeros((M,K))
    x = X_mat.flatten('F')

    # Build auxiliary matrices
    Pk_list = []
    for k in range(K):
        Pk = np.concatenate((np.zeros((M,M*k)), np.eye(M), np.zeros((M,M*(K-k-1)))), axis = 1 )
        Pk_list.append(Pk)
    Pm_list = []
    for m in range(M):
        Pm = np.zeros((K,M*K))
        for k in range(K):
            Pm[k,m+M*k] = 1
        Pk_list.append(Pm)

    one_vec = np.ones(M*K)

    # build A and b

    A_list = []
    b_list = []

    b_tilde_list = []
    A_tilde_list = []

    for k in range(K):
        A_tilde = np.zeros((M,M))
        b_tilde = np.zeros(M)

        for m in range(M):
            for n in range(M):
                A_tilde[m,n] = rho*eta[k]*alpha[m,k]*alpha[n,k]
            b_tilde[m] = alpha[m,k]*(rho*(np.sum(np.multiply(eta,beta[m,:])) - beta[m,k]*eta[k]) + 1)

        A_tilde_list.append(A_tilde)
        b_tilde_list.append(b_tilde)

        A_list.append( Pk_list[k].T@A_tilde@Pk_list[k] )
        b_list.append( Pk_list[k].T @ b_tilde )



    A_lambda = np.zeros((M*K,M*K))
    b_lambda = np.zeros(M*K)
    for k in range(K):
        A_lambda += lambda_s[k] * A_list[k]
        b_lambda += t*lambda_s[k]*b_list[k]


    # Optimize
    print('----- Starting QAOA -----')

    expectation = get_expectation_fixedt(M,K,BW_max, A_lambda, b_lambda, p_param)

    theta = np.array((2*p_param) * [1/p_param])
    # theta_extended = np.array( theta + list(map(lambda x:pow(10,x),lambda_s)) )
    #theta_extended = np.array( theta + lambda_s)

    res = minimize(expectation,
                          theta,
                          method='COBYLA')


    backend = Aer.get_backend('aer_simulator')
    backend.shots = 1024

    print('COBYLA optimization result: ', res.x)

    qc_res = create_qaoa_circ_fixedt(M,K,BW_max, A_lambda, b_lambda, res.x)
    qc_res = transpile(qc_res, backend)
    counts = backend.run(qc_res, seed_simulator=2703, shots = 1024).result().get_counts()

    # Return variables
    # Find best X

    x_best = max(counts, key=counts.get)
    v_best = counts[x_best]
    print('v best = ', v_best)
    #xbest_list = []
    #[xbest_list.append(k) for k,v in counts.items() if float(v) >= 0.66*v_best]

    #xbest_list_sorted = best_n_SINR(xbest_list, M, K, eta, alpha, beta, rho)
    #x_bitstring = xbest_list_sorted[0]
    #X_opt = bitstring_to_mat(x_bitstring, M, K)
    #X_opt = bitstring_to_mat(x_best, M, K)
    # convert xbest_bitstring to binary vector

    # Find average X

    X_opt = np.zeros((M,K))
    avgSINR = 0
    avgconvalues = np.zeros(K)
    sum_count = 0
    avgconnect = 0
    avgobj = 0
    for elem,count in counts.items():
        Xmat = bitstring_to_mat(elem, M, K)
        X_opt += Xmat*count
        obj, nconstr, convalues = check_soft_contraints(Xmat.flatten('F'), t,A_list, b_list, lambda_s)
        avgconnect += nconstr
        avgobj += obj
        avgSINR += (SINR(Xmat, eta, alpha, beta, rho) * count)
        avgconvalues += convalues
        sum_count += count
    X_opt = X_opt/sum_count
    avgSINR = avgSINR/sum_count
    avgobj = avgobj/sum_count
    avgconnect = avgconnect/sum_count
    avgconvalues = avgconvalues/sum_count

    X_opt = bitstring_to_mat(x_best, M, K) # comment to take the avg, uncomment to take MAP
    return X_opt, avgSINR, avgconnect, avgconvalues, counts




def optimize_links(BW_max, rho, eta, alpha, beta, t,  lambda_0, tg, p_param = 2, ):
    # optimize network links using QAOA algorithm
    # BW_max: max bandwidth of network
    # rho: max power for any user
    # eta: [vector of dimension K] power control for each user
    # alpha, beta: [matrices M x K] network parameters (see notes)
    # t: auxiliary variables - current optimal snr
    # p_param: depth of the QAOA circuit
    # lambda_s: [vector of dim K] set of penalization parameters for soft contraints (see notes)

    M, K = beta.shape[0], beta.shape[1]
    #tg = 10.0 ** np.array([ -3, -2.5, -2.2, -1.8 , -1.5, -1, -0.5]) # should find a good way to produce this
    #tg = 10.0 ** np.array([ -3, -2.8 , -2.5, -2.3 , -2.2, -2.0 , -1.8, -1.6 , -1.5, -1, -0.5]) # should find a good way to produce this
    #tg = np.array([0.01, 0.43, 0.4345440739052959, 0.44])
    Nt = len(tg)
    x_t = np.zeros(Nt)
    X_mat = np.zeros((M,K))
    x_vec = X_mat.flatten('F')
    x = np.concatenate((x_vec,x_t))

    # Build auxiliary matrices
    Pt = np.concatenate((np.zeros((Nt,M*K)),np.eye(Nt)), axis = 1)
    Pk_list = []
    for k in range(K):
        Pk = np.concatenate((np.zeros((M,M*k)), np.eye(M), np.zeros((M,M*(K-k-1)+Nt))), axis = 1 )
        Pk_list.append(Pk)
    Pm_list = []
    for m in range(M):
        Pm = np.zeros((K,M*K+Nt))
        for k in range(K):
            Pm[k,m+M*k] = 1
        Pk_list.append(Pm)

    one_vec = np.ones(M*K+Nt)
    one_vec_tilde = np.concatenate((np.ones(M*K),np.zeros(Nt)))

    # build A and B

    A_list = []
    B_list = []
    C_list = []
    cbar_list = []
    ctilde_list = []

    b_tilde_list = []
    A_tilde_list = []

    for k in range(K):
        A_tilde = np.zeros((M,M))
        b_tilde = np.zeros(M)

        for m in range(M):
            for n in range(M):
                A_tilde[m,n] = rho*eta[k]*alpha[m,k]*alpha[n,k]
            b_tilde[m] = alpha[m,k]*(rho*(np.sum(np.multiply(eta,beta[m,:])) - beta[m,k]*eta[k]) + 1)

        A_tilde_list.append(A_tilde)
        b_tilde_list.append(b_tilde)

        A_list.append( Pk_list[k].T@A_tilde@Pk_list[k] )
        B_list.append( np.outer((Pt.T @ tg),(b_tilde.T @ Pk_list[k])) )

        C_list.append( A_list[k] - B_list[k])
        cbar_list.append( np.sum(C_list[k],0) )
        ctilde_list.append( np.sum(C_list[k],1) )

    for k in range(K):
        for i in range(M*K+Nt):
            for j in range(M*K+Nt):
                C_list[k][i,j] = min(0,C_list[k][i,j])

    #C = np.sum( [ A_list[k] - B_list[k] for k in range(len(A_list)) ] , 0 )

    #c_bar = np.sum(C,0)
    #c_tilde = np.sum(C,1)

    lambda_s = np.zeros(K)
    for k in range(K):
        lambda_s[k] = lambda_0/max([np.trace(C_list[k]), 10**(-3) ])

    # Optimize
    print('----- Starting QAOA -----')

    expectation = get_expectation(M,K,BW_max,Nt, Pt, tg, C_list, cbar_list, ctilde_list, p_param, lambda_s)

    theta = np.array((2*p_param) * [1/p_param])
    # theta_extended = np.array( theta + list(map(lambda x:pow(10,x),lambda_s)) )
    #theta_extended = np.array( theta + lambda_s)

    res = minimize(expectation,
                          theta,
                          method='COBYLA')


    backend = Aer.get_backend('aer_simulator')
    backend.shots = 8192

    print('COBYLA optimization result: ', res.x)

    qc_res = create_qaoa_circ(M,K,BW_max,Nt, tg, C_list, cbar_list, ctilde_list, res.x, lambda_s)
    qc_res = transpile(qc_res, backend)
    counts = backend.run(qc_res, seed_simulator=2703, shots = 8192).result().get_counts()


    # Return variables
    # Find best X

    x_best = max(counts, key=counts.get)
    v_best = counts[x_best]
    print('v best = ', v_best)
    xbest_list = []
    [xbest_list.append(k) for k,v in counts.items() if float(v) >= 0.66*v_best]
    #print('Identified ',len(xbest_list),' combinations within 75% of best value:\n')
    #print(xbest_list)
    '''
    xv_best_list = []
    for x in xbest_list:
        xv = np.zeros(len(x))
        for i in range(len(x)):
            if x[i] == '1':
                xv[i] = 1
        xv_best_list.append(xv[::-1])
        '''
    xbest_list_sorted = best_n_SINR(xbest_list, M, K, eta, alpha, beta, rho)
    x_bitstring = xbest_list_sorted[0]
    # convert xbest_bitstring to binary vector

    xbv = np.zeros(len(x_bitstring))
    for i in range(len(x_bitstring)):
        if x_bitstring[i] == '1':
            xbv[i] = 1

    xbv = xbv[::-1] # invert MSB - LSB
    # Assign best solution to X_mat
    X_mat = xbv[:M*K].reshape((M,K))

    # Find best t
    #t = xbv[M*K:].dot(tg)
    t = min( SINR( X_mat, eta, alpha, beta, rho ) )

    if sum(xbv[M*K:]) == 0:
        print('>>>> QAOA finds t = 0, while real is t = ', t, ' and min tg = ', min(tg))

    print('Optimal t = ', t)
    print('Optimal X = ', X_mat)

    return X_mat, t, xbv[M*K:]

def optimize_powers(rho, X, alpha, beta, t, tol = 10**(-7) , maxiter = 100):
    # optimize user powers via Bisection method
    K = beta.shape[1]
    eta = cp.Variable(K, nonneg = True)
    G = cp.Parameter((K,K))
    d = cp.Parameter(K)
    prob = cp.Problem(cp.Minimize(0),
                        [G@eta - d >= 0,
                        eta <= np.ones(K)])

    tmin = t/10
    tmax = 10*t
    it = 0
    print('----- Starting Bisection Method -----')
    while (tmax-tmin)>tol and it < maxiter :
        it = it+1
        t = (tmin + tmax)/2


        G.value, d.value = initialize_feas_params(rho, X, alpha, beta, t)


        prob.solve()
        #print('Status: ', prob.status)
        if prob.status=='infeasible':
            tmax = t
        else:
            tmin = t

    #print('D E B U G   --->   ', rho, X, alpha, beta, tmin)
    G.value, d.value = initialize_feas_params(rho, X, alpha, beta, tmin)
    prob.solve()
    eta_opt = prob.variables()

    print('Optimal t = ',tmin)
    print('Optimal eta = ', eta_opt[0].value)

    return eta_opt[0].value, tmin

# ---------------- Bisection subroutines: ---------------------

def initialize_feas_params(rho, X, alpha, beta, t):
    # initialize bisection parameters
    K = len(alpha[1,:])
    dtmp = np.zeros(K)
    Gtmp = np.zeros((K,K))

    for k in range(K): # for row
        dtmp[k] = X[:,k].dot(alpha[:,k])
        Gtmp[k,k] = rho/t * dtmp[k]**2
        for k1 in range(K): # for column (excluding diagonal)
            if k1 != k:
                Gtmp[k,k1] = -rho * X[:,k].dot( np.multiply(alpha[:,k],beta[:,k1]) )


    return [Gtmp,dtmp]

# ------------------ QAOA subroutines: ------------------------

def problem_objective_fixedt(x, A_lambda, b_lambda, verbose = False):

    # convert x to binary vector
    xv = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] == '1':
            xv[i] = 1

    xv = xv[::-1]

    obj = (xv.T @ A_lambda - b_lambda.T) @ xv

    return(-obj)  # we minimize


def problem_objective(x, tg, Pt, C_list, lambda_soft, verbose = False):

    K = len(lambda_soft)

    # convert x to binary vector
    xv = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] == '1':
            xv[i] = 1

    xv = xv[::-1]

    pen = 0
    for k in range(K):
        pen += lambda_soft[k]* xv.T @ (C_list[k] @ xv)
        #print(k,'-th term in penalization : ',lambda_soft[k]* xv.T @ (C_list[k] @ xv))


    if verbose:
        return(tg.T @ (Pt @ xv), pen)

    obj = tg.T @ (Pt @ xv) + pen

    return(-obj)  # we minimize


def compute_expectation_fixedt(counts, A_lambda, b_lambda):

    """
    Computes expectation value based on measurement results

    """

    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():

        obj = problem_objective_fixedt(bitstring, A_lambda, b_lambda)
        avg += obj * count
        sum_count += count

    return avg/sum_count

def compute_expectation(counts, tg, Pt, C_list, lambda_soft):

    """
    Computes expectation value based on measurement results

    """

    avg = 0
    sum_count = 0
    for bitstring, count in counts.items():

        obj = problem_objective(bitstring, tg, Pt, C_list, lambda_soft)
        avg += obj * count
        sum_count += count

    return avg/sum_count


def create_qaoa_circ_fixedt(M,K,BW_max, A_lambda, b_lambda, theta):

    """
    Creates a parametrized qaoa circuit
    """

    nqubits = M*K
    MK = M*K

    p = len(theta) //2  # number of alternating unitaries

    beta = theta[:p]
    gamma = theta[p:]
    # lambda_soft = list(map(lambda x:pow(10,x), theta_extended[2*p:] ))
    #lambda_soft =  theta_extended[2*p:]

    A_sum = (A_lambda + A_lambda.T)@ np.ones(MK)

    # initial_state
    qc = QuantumCircuit(nqubits)
    Us = QAOAut.U_alpha(MK,BW_max) +  QAOAut.Unn(MK)
    qc.compose(Us,qubits = list(range(MK)), inplace = True)

    # phase separation unitary
    for irep in range(0, p):

        # problem unitary
        for i in range(MK):
            for j in range(MK):
                coef = A_lambda[i,j].item()
                if(i != j and abs(coef)>10**(-12)):
                    qc.rzz( -0.25 * coef * gamma[irep].item(), i, j)


        for i in range(MK):
            coef = 0.25 * A_sum[i] - 0.5 * b_lambda[i]
            if(abs(coef) > 10**(-12)):
                qc.rz( gamma[irep].item() * coef, i)

        # mixing unitary
        Um_ring = QAOAut.Um_ring(MK,beta[irep])
        #print(Um_ring)
        qc.compose(Um_ring,qubits = list(range(MK)), inplace = True)

    qc.measure_all()

    return qc


def create_qaoa_circ(M,K,BW_max,Nt, tg, C_list, cbar_list, ctilde_list, theta, lambda_soft):

    """
    Creates a parametrized qaoa circuit

    """
    nqubits = M*K+Nt
    MK = M*K

    p = len(theta) //2  # number of alternating unitaries

    beta = theta[:p]
    gamma = theta[p:]
    # lambda_soft = list(map(lambda x:pow(10,x), theta_extended[2*p:] ))
    #lambda_soft =  theta_extended[2*p:]

    # initial_state
    qc = QuantumCircuit(nqubits)
    Us = QAOAut.U_alpha(MK,BW_max) +  QAOAut.Unn(MK)
    qc.compose(Us,qubits = list(range(MK)), inplace = True)
    for i in range(MK, nqubits):
        qc.h(i)



    # phase separation unitary
    for irep in range(0, p):

        # problem unitary
        for i in range(M*K+Nt):
            for j in range(M*K+Nt):
                coef = 0
                for k in range(K):
                    C = C_list[k]
                    coef += lambda_soft[k] * C[i,j].item()
                if(i != j and abs(coef)>10**(-12)):
                    qc.rzz( -0.25 * coef * gamma[irep].item(), i, j)


        for i in range(M*K+Nt):
            coef = 0
            for k in range(K):
                coef += 0.25 * lambda_soft[k]* (ctilde_list[k][i] + cbar_list[k][i])
            if(i > M*K):
                coef += 0.5*tg[i-M*K]
            if(abs(coef) > 10**(-12)):
                qc.rz( gamma[irep].item() * coef, i)

        # mixing unitary
        Um_ring = QAOAut.Um_ring(MK,beta[irep])
        #print(Um_ring)
        qc.compose(Um_ring,qubits = list(range(MK)), inplace = True)
        for i in range(MK, nqubits):
            qc.rx(2 * beta[irep], i)

    qc.measure_all()

    return qc


def get_expectation_fixedt(M,K,BW_max, A_lambda, b_lambda, p, shots=4096):

    """
    Runs parametrized circuit
    """

    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots

    def execute_circ(theta):
        # lambda_soft = list(map(lambda x:pow(10,x),theta_extended[2*p:]))
        qc = create_qaoa_circ_fixedt(M,K,BW_max, A_lambda, b_lambda, theta)
        qc = transpile(qc, backend)
        counts = backend.run(qc, seed_simulator=2703, nshots=shots).result().get_counts()

        return compute_expectation_fixedt(counts, A_lambda, b_lambda)

    return execute_circ


def get_expectation(M,K,BW_max,Nt, Pt, tg, C_list, cbar_list, ctilde_list, p, lambda_s, shots=4096):

    """
    Runs parametrized circuit
    """

    backend = Aer.get_backend('qasm_simulator')
    backend.shots = shots

    def execute_circ(theta):
        # lambda_soft = list(map(lambda x:pow(10,x),theta_extended[2*p:]))
        qc = create_qaoa_circ(M,K,BW_max,Nt, tg, C_list, cbar_list, ctilde_list, theta, lambda_s)
        qc = transpile(qc, backend)
        counts = backend.run(qc, seed_simulator=2703, nshots=shots).result().get_counts()

        return compute_expectation(counts, tg, Pt, C_list, lambda_s)

    return execute_circ


# -------------- General Cell-Free MIMO functions: -------------------

def SINR( X, eta, alpha, beta, rho ):

    K = X.shape[1]
    M = X.shape[0]

    snr = np.zeros(K)

    for k in range(K):
        num = rho * eta[k]*( np.sum(np.multiply(X[:,k],alpha[:,k]))**2 )
        den = 0
        for k1 in list(range(k))+list(range(k+1,K)):
            den += rho*( eta[k1]* ( np.sum( np.multiply(np.multiply(X[:,k],alpha[:,k]), beta[:,k1] ) ) ) )
        den += np.sum(np.multiply(X[:,k],alpha[:,k]))
        if abs(den)>10**(-10):
            snr[k] = num/den
        else:
            snr[k] = 0
        if snr[k] != snr[k] : snr[k] = 0

    return snr



def SINR_from_bitstring(x, M, K, eta, alpha, beta, rho):

    # convert bitstring to matrix and call SINR
    xv = np.zeros(len(x))
    for i in range(len(x)):
        if x[i] == '1':
            xv[i] = 1

    xv = np.array(xv[::-1])

    X_mat = xv[:M*K].reshape((M,K),order='F')

    return(SINR( X_mat, eta, alpha, beta, rho ))


def best_n_SINR(bitstring_list, M, K, eta, alpha, beta, rho):
    # Given a list of solutions (network links) order them by best SINR ( max min SINR )
    bitstring_list.sort(reverse=True, key = lambda x: min(SINR_from_bitstring(x, M, K, eta, alpha, beta, rho)))

    return(bitstring_list)


def bruteforce_SINR(M, K, BW_max, eta, alpha, beta, rho):
    # find best SINR by bruteforce search
    print('----- Bruteforce Solution -----')
    n_combin = 2**(M*K)

    X_list_try = []

    for i in range(n_combin):
        x_vec = np.zeros(M*K)
        bs = format(i, "0"+str(M*K)+"b")
        for j in range(M*K):
            x_vec[j] = bs[j]
        X_list_try.append(x_vec.reshape(M,K))

    X_list = []
    for Xmat in X_list_try:
        if np.sum(Xmat.flatten()) <= BW_max:
            X_list.append(Xmat)

    min_snr = np.zeros(len(X_list))
    for i in range(len(X_list)):
        Xmat = X_list[i]
        snr = SINR(Xmat, eta, alpha, beta, rho)
        min_snr[i] = np.amin(snr)

    best_id = np.argmax(min_snr)
    best_X = X_list[best_id]

    #print('Actual full search best\n')
    #print('ID: ', best_id)
    print('MIN SINR: ', min_snr[best_id])
    print('Best X: ', best_X)
    bestsinrs = np.sort(min_snr)
    print('Best 10 SINRs: ', bestsinrs[-10:] )


    return(best_X, min_snr[best_id])


def bitstring_to_mat(x_bitstring, M, K):

    xbv = np.zeros(len(x_bitstring))
    for i in range(len(x_bitstring)):
        if x_bitstring[i] == '1':
            xbv[i] = 1

    xbv = xbv[::-1] # invert MSB - LSB
    # Assign best solution to X_mat
    X_mat = xbv.reshape((M,K))

    return X_mat

#------------------- Others ----------------------------

def check_soft_contraints(xv, t, A_list, b_list, lambda_s):

    K = len(lambda_s)
    M = len(xv)//K

    A_lambda = np.zeros((M*K,M*K))
    b_lambda = np.zeros(M*K)
    for k in range(K):
        A_lambda += lambda_s[k] * A_list[k]
        b_lambda += t*lambda_s[k]*b_list[k]

    obj = (xv.T @ A_lambda - b_lambda.T) @ xv
    ccon = 0
    convalues = np.zeros(K)
    constraint_state = np.ones(K)
    for k in range(K):
        convalues[k] = (xv.T@A_list[k] - t*b_list[k])@xv
        ccon += convalues[k]
        if(convalues[k] < 0):
            constraint_state[k]=0
    return obj, sum(constraint_state), convalues
