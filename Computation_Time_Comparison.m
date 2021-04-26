clear
close all
clc

%% Seed
%rng( 3 );

%% cycle initiation
max_rep = 10;
max_size = 300;
time_LG_parameters = zeros( max_size , 1 );
time_LG_num_entropy = zeros( max_size, 1 );
time_LG_upper_bound_entropy = zeros( max_size, 1 );
time_dir_parameters = zeros( max_size , 1 );
time_dir_entropy = zeros( max_size , 1 );

dir_entropy = zeros( max_size , 1 );
lg_entropy_num = zeros( max_size , 1 );
lg_entropy_upper = zeros( max_size , 1 );
lg_entropy_lower = zeros( max_size , 1 );

log_likelihood_lg = zeros( max_size , 1 );
log_likelihood_dir = zeros( max_size , 1 );

for rep = 1 : 1 : max_rep
for cycle = 2 : 1 : max_size

%% Initialization 
CPV_number = 1000;
CPV_dimension = cycle;

expectation = 5 - 10 * rand( CPV_dimension - 1 , 1 );
variances_diag = 2 * rand( CPV_dimension - 1 , 1 );
covariance = diag( variances_diag );

samples = mvnrnd( expectation , covariance , CPV_number );
samples( : , end+1 ) = 0;
samples_CPV = zeros( CPV_number , CPV_dimension );

for idx = 1 : 1 : CPV_number
    samples_CPV( idx , : ) = exp( samples( idx , : ) ) ./ sum( exp( samples( idx , : ) ) );
end

%% Logistical Gaussian parameter computation
lg_samples = zeros( CPV_number , CPV_dimension );

tic %Start LG inference clock

for idx = 1 : 1 : CPV_number
    lg_samples( idx , : ) = log( samples_CPV( idx , : ) ) - log( samples_CPV( idx , end ) );
end

lg_samples = lg_samples( : , 1 : end - 1 );
lg_expectation = sum( lg_samples , 1 ) / CPV_number;
lg_covariance = zeros( CPV_dimension - 1 , CPV_dimension - 1 );

for idx = 1 : 1 : CPV_number
    
    lg_covariance = lg_covariance + ( lg_samples( idx , : ) - lg_expectation )'...
        * ( lg_samples( idx , : ) - lg_expectation ) / CPV_number;
    
end

time_LG_parameters_temp = toc; %Stop LG inference clock
time_LG_parameters( cycle )  = time_LG_parameters( cycle ) + time_LG_parameters_temp / max_rep;

%% Logistical Gaussian numerical entropy

tic %Start LG numerical entropy

lg_pdf_value = mvnpdf( lg_samples , lg_expectation , lg_covariance + eye( CPV_dimension - 1 ) * 1e-10 );

lg_log_pdf_value = log( lg_pdf_value );

for idx = 1 : 1 : CPV_number
    lg_log_pdf_value( idx ) = lg_log_pdf_value( idx ) - sum( log( samples_CPV( idx , : ) ) );
end

lg_entropy_num_temp = - sum( lg_log_pdf_value ) / CPV_number;
lg_entropy_num( cycle ) = lg_entropy_num( cycle ) + lg_entropy_num_temp / max_rep;

time_LG_num_entropy_temp = toc; %Stop LG numerical entropy
time_LG_num_entropy( cycle ) = time_LG_num_entropy( cycle ) + time_LG_num_entropy_temp / max_rep;

log_likelihood_lg_temp = sum( lg_log_pdf_value );
log_likelihood_lg( cycle ) = log_likelihood_lg( cycle ) + log_likelihood_lg_temp / max_rep / CPV_number;

%% Logistical Gaussian bound entropy

tic %Start LG bound

lg_ent_initial = 0.5 * log( 2 * pi * exp( 1 ) * det( lg_covariance ) );

lg_entropy_upper_temp = lg_ent_initial + sum( lg_expectation ) - CPV_dimension * max( [ 0 , lg_expectation ] );
lg_entropy_upper( cycle ) = lg_entropy_upper( cycle ) + lg_entropy_upper_temp / max_rep;
lg_entropy_lower( cycle ) = lg_entropy_lower( cycle ) +...
    ( lg_entropy_upper_temp - CPV_dimension * log( CPV_dimension ) ) / max_rep;

time_LG_upper_bound_entropy_temp = toc; % Stop LG bound
time_LG_upper_bound_entropy( cycle )= time_LG_upper_bound_entropy( cycle )...
    + time_LG_upper_bound_entropy_temp / max_rep;

%% Dirichlet parameter computation

tic %Start Dir inference

%log_CPV = log( samples_CPV );
%log_CPV_avg = sum( log_CPV , 1 ) / CPV_number;

log_CPV = log( samples_CPV );
log_CPV_avg = sum( log_CPV , 1 ) / CPV_number;

alpha = ones( CPV_dimension , 1 );
difference = 100;
%for iteration = 1 : 1 : 50
while difference >= 1e-6
    alpha_temp = invpsi( psi( sum( alpha ) ) + log_CPV_avg );
    difference = sum( abs( alpha_temp - alpha ) );
    alpha = alpha_temp;
end

time_dir_parameters_temp = toc; %Stop Dir inference
time_dir_parameters( cycle ) = time_dir_parameters( cycle ) + time_dir_parameters_temp / max_rep;

log_likelihood_dir_temp = 0;
for idx = 1 : 1 : CPV_number
    log_likelihood_dir_temp_temp = + gammaln( sum( alpha ) );
    for idx_class = 1 : 1 : CPV_dimension
        log_likelihood_dir_temp_temp = log_likelihood_dir_temp_temp - gammaln( alpha( idx_class ) )...
            + ( alpha( idx_class ) - 1 ) * log( samples_CPV( idx , idx_class ) );
    end
    log_likelihood_dir_temp  = log_likelihood_dir_temp + log_likelihood_dir_temp_temp;
end
log_likelihood_dir( cycle ) = log_likelihood_dir( cycle ) + log_likelihood_dir_temp / max_rep / CPV_number;

%% Dirichlet entropy

tic %Start Dir entropy

dir_entropy_temp = - gammaln( sum( alpha ) ) + ( sum( alpha ) - CPV_dimension ) * psi( sum( alpha ) );
for idx = 1 : 1 : CPV_dimension
    dir_entropy_temp = dir_entropy_temp + gammaln( alpha( idx ) ) - ( alpha( idx ) - 1 ) * psi( alpha( idx ) ); 
end
dir_entropy( cycle ) = dir_entropy( cycle ) + dir_entropy_temp / max_rep;

time_dir_entropy_temp = toc; %Stop Dir entropy
time_dir_entropy( cycle ) = time_dir_entropy( cycle ) + time_dir_entropy_temp / max_rep;

end
end

%% Plots
figure('Name' , 'Inference time')
grid on
hold on
plot(time_dir_parameters, 'LineWidth', 2)
plot(time_LG_parameters, 'LineWidth', 2)
set( gca , 'FontSize' , 18)
legend('Dirichlet' , 'LG')
xlabel('Number of candidate classes')
ylabel('Computational time [s]')
hold off

figure('Name' , 'Entropy computation time')
grid on
hold on
plot(time_dir_entropy, 'LineWidth', 2)
plot(time_LG_num_entropy, 'LineWidth', 2)
plot(time_LG_upper_bound_entropy, 'LineWidth', 2)
set( gca , 'FontSize' , 18)
legend('Dirichlet' , 'LG numerical' , 'LG upper bound')
xlabel('Number of candidate classes')
ylabel('Computational time [s]')
hold off

figure('Name' , 'Entropy value')
grid on
hold on
plot(dir_entropy, 'LineWidth', 2)
plot(lg_entropy_num, 'LineWidth', 2)
plot(lg_entropy_upper, 'LineWidth', 2)
plot(lg_entropy_lower, 'LineWidth', 2)
set( gca , 'FontSize' , 18)
legend('Dirichlet' , 'LG numerical' , 'LG upper bound', 'LG lower bound')
xlabel('Number of candidate classes')
ylabel('Entropy value')
hold off

figure('Name' , 'Log likelihood graph')
grid on
hold on
plot(log_likelihood_dir, 'LineWidth', 2)
plot(log_likelihood_lg, 'LineWidth', 2)
set( gca , 'FontSize' , 18)
legend('Dirichlet' , 'LG')
xlabel('Number of candidate classes')
ylabel('Average log likelihood')
hold off

return;