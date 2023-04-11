syms z A B C D
alpha_AC = .7;
alpha_AD = .3;
alpha_BC = .2;
alpha_BD = .8;
beta_AC = floor(.1*Fs);
beta_AD = floor(.3*Fs);
beta_BC = floor(.7*Fs);
beta_BD = floor(.01*Fs);

eq1 = alpha_AC*z^
