clear
close all
clc

%%  Initializing
expectation = 5;
variance = 2;
num_samples = 30;
var_range = 0.8;

%% COmputing
samples = normrnd( expectation, variance , [1,num_samples] );
x = ones( 1 , num_samples );
x_adj = [2 3 4 5];
avg = sum( samples ) / num_samples;
samples_sorted = sort( samples );
var = samples_sorted( round( ( 1 - var_range ) * num_samples ) );
cvar = sum( samples_sorted( 1 : round( ( 1 - var_range ) * num_samples ) ) ) /...
    ( round( ( 1 - var_range ) * num_samples ) );
worst_case = samples_sorted( 1 );

%% Plot
figure('Name', 'CVaR demonstration')
hold on
grid on
plot( x( 1 : num_samples ) , samples  ,'*' );
bar( x_adj , [ avg , var , cvar , worst_case ] )
labels = {'Expectation' , 'VaR' , 'CVaR' , 'Worst Case' };
text(2:5, [ avg , var , cvar , worst_case ], labels, 'HorizontalAlignment','center', 'VerticalAlignment','bottom','FontSize' , 18)
set( gca , 'FontSize' , 18 ,'xtick', [])
ylabel('Reward value')
axis()
hold off

return;