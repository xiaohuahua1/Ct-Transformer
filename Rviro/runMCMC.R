R0 <- 3.2
net <- "SF"

if(net == "ER" & R0 == 1.2)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=1.2\\test\\ctData\\ctDataID15.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=1.2\\test\\ctData\\Rt.txt", header = FALSE)
}
if(net == "ER" & R0 == 1.4)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=1.4\\test\\ctData\\ctDataID9.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=1.4\\test\\ctData\\Rt.txt", header = FALSE)
}
if(net == "ER" & R0 == 1.6)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=1.6\\test\\ctData\\ctDataID13.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=1.6\\test\\ctData\\Rt.txt", header = FALSE)
}
if(net == "ER" & R0 == 1.8)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=1.8\\test\\ctData\\ctDataID14.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=1.8\\test\\ctData\\Rt.txt", header = FALSE)
}
if(net == "ER" & R0 == 2.2)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=2.2\\test\\ctData\\ctDataID5.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=2.2\\test\\ctData\\Rt.txt", header = FALSE)
}
if(net == "ER" & R0 == 2.4)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=2.4\\test\\ctData\\ctDataID15.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=2.4\\test\\ctData\\Rt.txt", header = FALSE)
}
if(net == "ER" & R0 == 2.6)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=2.6\\test\\ctData\\ctDataID20.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=2.6\\test\\ctData\\Rt.txt", header = FALSE)
}
if(net == "ER" & R0 == 2.8)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=2.8\\test\\ctData\\ctDataID11.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=2.8\\test\\ctData\\Rt.txt", header = FALSE)
}
if(net == "ER" & R0 == 3.2)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=3.2\\test\\ctData\\ctDataID18.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=3.2\\test\\ctData\\Rt.txt", header = FALSE)
}
if(net == "ER" & R0 == 3.4)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=3.4\\test\\ctData\\ctDataID14.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=3.4\\test\\ctData\\Rt.txt", header = FALSE)
  SEIRData <- read.csv("..\\results\\ER\\d=10R=3.4\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 1.2)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=1.2\\test\\ctData\\ctDataID7.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=1.2\\test\\ctData\\Rt.txt", header = FALSE)
  SEIRData <- read.csv("..\\results\\SF\\d=10R=1.2\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 1.4)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=1.4\\test\\ctData\\ctDataID25.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=1.4\\test\\ctData\\Rt.txt", header = FALSE)
  SEIRData <- read.csv("..\\results\\SF\\d=10R=1.4\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 1.6)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=1.6\\test\\ctData\\ctDataID12.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=1.6\\test\\ctData\\Rt.txt", header = FALSE)
  SEIRData <- read.csv("..\\results\\SF\\d=10R=1.6\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 1.8)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=1.8\\test\\ctData\\ctDataID22.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=1.8\\test\\ctData\\Rt.txt", header = FALSE)
  SEIRData <- read.csv("..\\results\\SF\\d=10R=1.8\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 2.2)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=2.2\\test\\ctData\\ctDataID2.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=2.2\\test\\ctData\\Rt.txt", header = FALSE)
  SEIRData <- read.csv("..\\results\\SF\\d=10R=2.2\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 2.4)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=2.4\\test\\ctData\\ctDataID29.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=2.4\\test\\ctData\\Rt.txt", header = FALSE)
  SEIRData <- read.csv("..\\results\\SF\\d=10R=2.4\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 2.6)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=2.6\\test\\ctData\\ctDataID17.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=2.6\\test\\ctData\\Rt.txt", header = FALSE)
  SEIRData <- read.csv("..\\results\\SF\\d=10R=2.6\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 2.8)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=2.8\\test\\ctData\\ctDataID13.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=2.8\\test\\ctData\\Rt.txt", header = FALSE)
  SEIRData <- read.csv("..\\results\\SF\\d=10R=2.8\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 3.2)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=3.2\\test\\ctData\\ctDataID15.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=3.2\\test\\ctData\\Rt.txt", header = FALSE)
  SEIRData <- read.csv("..\\results\\SF\\d=10R=3.2\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 3.4)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=3.4\\test\\ctData\\ctDataID13.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=3.4\\test\\ctData\\Rt.txt", header = FALSE)
  SEIRData <- read.csv("..\\results\\SF\\d=10R=3.4\\test\\ctData\\SEIR_partab.csv")
}



RtDatat <- t(RtData)
RtDatat <- as.data.frame(RtDatat)
num_rows <- nrow(RtDatat)
times_Rt <- 1:num_rows
times <- as.data.frame(times_Rt)


mat <- matrix(rep(times_Rt, each=length(times_Rt)),ncol=length(times_Rt))
t_dist <- abs(apply(mat, 2, function(x) x-times_Rt))

par_tab <- bind_rows(SEIRData[SEIRData$names != "prob",], SEIRData[SEIRData$names == "prob",][1:length(times),])
pars <- par_tab$values
names(pars) <- par_tab$names
means <- par_tab$values
names(means) <- par_tab$names

sds_seir <- c("beta"=0.25,"R0"=0.6,
              "obs_sd"=0.5,"viral_peak"=2,
              "wane_rate2"=1,"t_switch"=3,"level_switch"=1,
              "prob_detect"=0.03,
              "incubation"=0.25, "infectious"=0.5,
              "rho"=2,"nu"=0.5)

prior_func_hinge_seir <- function(pars,...){
  obs_sd_prior <- dnorm(pars["obs_sd"], means[which(names(means) == "obs_sd")], sds_seir["obs_sd"],log=TRUE)
  #r0_prior <- dlnorm(pars["R0"],log(2),sds_seir["R0"],log=TRUE)
  viral_peak_prior <- dnorm(pars["viral_peak"], means[which(names(means) == "viral_peak")], sds_seir["viral_peak"],log=TRUE)
  wane_2_prior <- dnorm(pars["wane_rate2"],means[which(names(means) == "wane_rate2")],sds_seir["wane_rate2"],log=TRUE)
  tswitch_prior <- dnorm(pars["t_switch"],means[which(names(means) == "t_switch")],sds_seir["t_switch"],log=TRUE)
  level_prior <- dnorm(pars["level_switch"],means[which(names(means) == "level_switch")],sds_seir["level_switch"],log=TRUE)
  beta1_mean <- means[which(names(means) == "prob_detect")]
  beta1_sd <- sds_seir["prob_detect"]
  beta_alpha <- ((1-beta1_mean)/beta1_sd^2 - 1/beta1_mean)*beta1_mean^2
  beta_beta <- beta_alpha*(1/beta1_mean - 1)
  beta_prior <- dbeta(pars["prob_detect"],beta_alpha,beta_beta,log=TRUE)
  
  incu_prior <- dlnorm(pars["incubation"],log(means[which(names(means) == "incubation")]), sds_seir["incubation"], TRUE)
  infectious_prior <- dlnorm(pars["infectious"],log(means[which(names(means) == "infectious")]),sds_seir["infectious"],TRUE)
  
  obs_sd_prior + viral_peak_prior +
    wane_2_prior + tswitch_prior + level_prior + beta_prior +
    incu_prior + infectious_prior# + r0_prior
}

## MCMC chain options
mcmc_pars <- c("iterations"=15000,"popt"=0.44,"opt_freq"=1000,
               "thin"=50,"adaptive_period"=12000,"save_block"=1000)

output <- run_MCMC(parTab=par_tab,
                   data=ctData,
                   INCIDENCE_FUNC=solveSEIRModel_lsoda_wrapper,
                   PRIOR_FUNC=prior_func_hinge_seir,
                   mcmcPars=mcmc_pars,
                   filename="R0=3.2",
                   CREATE_POSTERIOR_FUNC=create_posterior_func,
                   mvrPars=NULL,
                   OPT_TUNING=0.2,
                   use_pos=FALSE,
                   t_dist=t_dist)
