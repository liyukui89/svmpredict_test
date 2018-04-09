#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <math.h>
#include "svm_0408.h"

struct svm_parameter Initialize_svm_parameter()  
{  
    struct svm_parameter svmpara;//svm的相关参数  
    svmpara.svm_type = C_SVC;//-s svm_type : set type of SVM (default 0)
    svmpara.kernel_type = RBF;  //matlab里默认的是：2  -t kernel_type : set type of kernel function (default 2)  RBF  LINEAR
    svmpara.degree = 3;  //-d degree : set degree in kernel function (default 3)
    svmpara.gamma = 0.2;  // 默认大小可选择特征的倒数 1/num_features，核函数中的gamma函数设置（只针对多项式/rbf/sigmoid核函数）  
    svmpara.coef0 = 0;  //-r coef0 : set coef0 in kernel function (default 0
    svmpara.nu = 0.5;  //-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
    svmpara.cache_size = 1;//缓存块大小 -m cachesize : set cache memory size in MB (default 100) 
    svmpara.C = 1;  //-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
    svmpara.eps = 1e-3;  //-e epsilon : set tolerance of termination criterion (default 0.001)
    svmpara.p = 0.1;  //-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
    svmpara.shrinking = 1;  //-h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
    svmpara.probability = 0;  //-b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
    svmpara.nr_weight = 0;  //wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
    svmpara.weight_label = NULL;  
    svmpara.weight = NULL;  
    return svmpara;  
}  

double k_function (const struct svm_node *x, const struct svm_node *y,const struct svm_parameter * param)
{
	double sum;
	double d ;
			
	switch(param->kernel_type)
	{
		
		case RBF:
		{
			 sum = 0;
			while(x->index != -1 && y->index !=-1)//y->index 一开始不对！
			{
				
				if(x->index == y->index)
				{
					 d = x->value - y->value;
					sum += d*d;
					++x;
					++y;
				}
				else
				{
					if(x->index > y->index)
					{	
						sum += y->value * y->value;
						++y;
					}
					else
					{
						sum += x->value * x->value;
						++x;
					}
				}
			}

			while(x->index != -1)
			{
				sum += x->value * x->value;
				++x;
			}

			while(y->index != -1)
			{
				sum += y->value * y->value;
				++y;
			}
			
			return exp(-param->gamma*sum);
		}
		
		default:
			return 0;  // Unreachable 
	}
}

double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values)
{
	int i,j,k;
	int p;
	//进else：
	int nr_class = model->nr_class;
	int l = model->l;
	double *kvalue = Malloc(double,l);
	int *start = Malloc(int,nr_class);
	int *vote = Malloc(int,nr_class);
	double sum = 0;
	int si,sj,ci,cj;
	double *coef1;
	double *coef2;
	int vote_max_idx;
	for(i=0;i<l;i++){
		kvalue[i] = k_function(x,model->SV[i],&(model->param));//here
		
	}
	
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+model->nSV[i-1];

	
	for(i=0;i<nr_class;i++)
		vote[i] = 0;
	 p=0;
		for(i=0;i<nr_class;i++)
			for( j=i+1;j<nr_class;j++)
			{
				
				 si = start[i];
				 sj = start[j];
				 ci = model->nSV[i];
				 cj = model->nSV[j];
				
			
				 coef1 = model->sv_coef[j-1];
				 coef2 = model->sv_coef[i];
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
				sum -= model->rho[p];
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		 vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
}


double svm_predict (const struct svm_model *model, const struct svm_node *x)
{
	int nr_class = model->nr_class;
	double *dec_values;
	double pred_result;
	//分类进else
	dec_values = Malloc(double, nr_class*(nr_class-1)/2);
	pred_result = svm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}
