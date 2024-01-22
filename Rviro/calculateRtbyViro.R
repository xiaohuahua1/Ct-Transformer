R0 <- 3.2
net <- "SF"

if(net == "ER" & R0 == 1.2)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=1.2\\test\\ctData\\ctDataID15.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=1.2\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\ER\\d=10R=1.2\\test\\ctData\\R0=1.2_univariate_chain.csv")
  file_path <- "..\\results\\ER\\d=10R=1.2\\test\\ctData\\viroRt.txt"
}
if(net == "ER" & R0 == 1.4)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=1.4\\test\\ctData\\ctDataID9.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=1.4\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\ER\\d=10R=1.4\\test\\ctData\\R0=1.4_univariate_chain.csv")
  file_path <- "..\\results\\ER\\d=10R=1.4\\test\\ctData\\viroRt.txt"
}
if(net == "ER" & R0 == 1.6)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=1.6\\test\\ctData\\ctDataID13.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=1.6\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\ER\\d=10R=1.6\\test\\ctData\\R0=1.6_univariate_chain.csv")
  file_path <- "..\\results\\ER\\d=10R=1.6\\test\\ctData\\viroRt.txt"
}
if(net == "ER" & R0 == 1.8)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=1.8\\test\\ctData\\ctDataID14.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=1.8\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\ER\\d=10R=1.8\\test\\ctData\\R0=1.8_univariate_chain.csv")
  file_path <- "..\\results\\ER\\d=10R=1.8\\test\\ctData\\viroRt.txt"
}
if(net == "ER" & R0 == 2.2)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=2.2\\test\\ctData\\ctDataID5.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=2.2\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\ER\\d=10R=2.2\\test\\ctData\\R0=2.2_univariate_chain.csv")
  file_path <- "..\\results\\ER\\d=10R=2.2\\test\\ctData\\viroRt.txt"
}
if(net == "ER" & R0 == 2.4)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=2.4\\test\\ctData\\ctDataID15.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=2.4\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\ER\\d=10R=2.4\\test\\ctData\\R0=2.4_univariate_chain.csv")
  file_path <- "..\\results\\ER\\d=10R=2.4\\test\\ctData\\viroRt.txt"
}
if(net == "ER" & R0 == 2.6)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=2.6\\test\\ctData\\ctDataID20.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=2.6\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\ER\\d=10R=2.6\\test\\ctData\\R0=2.6_univariate_chain.csv")
  file_path <- "..\\results\\ER\\d=10R=2.6\\test\\ctData\\viroRt.txt"
}
if(net == "ER" & R0 == 2.8)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=2.8\\test\\ctData\\ctDataID11.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=2.8\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\ER\\d=10R=2.8\\test\\ctData\\R0=2.8_univariate_chain.csv")
  file_path <- "..\\results\\ER\\d=10R=2.8\\test\\ctData\\viroRt.txt"
}
if(net == "ER" & R0 == 3.2)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=3.2\\test\\ctData\\ctDataID18.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=3.2\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\ER\\d=10R=3.2\\test\\ctData\\R0=3.2_univariate_chain.csv")
  file_path <- "..\\results\\ER\\d=10R=3.2\\test\\ctData\\viroRt.txt"
}
if(net == "ER" & R0 == 3.4)
{
  ctData <- read.csv("..\\results\\ER\\d=10R=3.4\\test\\ctData\\ctDataID14.csv")
  RtData <- read.table("..\\results\\ER\\d=10R=3.4\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\ER\\d=10R=3.4\\test\\ctData\\R0=3.4_univariate_chain.csv")
  file_path <- "..\\results\\ER\\d=10R=3.4\\test\\ctData\\viroRt.txt"
  SEIRData <- read.csv("..\\results\\ER\\d=10R=3.4\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 1.2)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=1.2\\test\\ctData\\ctDataID7.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=1.2\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\SF\\d=10R=1.2\\test\\ctData\\R0=1.2_univariate_chain.csv")
  file_path <- "..\\results\\SF\\d=10R=1.2\\test\\ctData\\viroRt.txt"
  SEIRData <- read.csv("..\\results\\SF\\d=10R=1.2\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 1.4)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=1.4\\test\\ctData\\ctDataID25.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=1.4\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\SF\\d=10R=1.4\\test\\ctData\\R0=1.4_univariate_chain.csv")
  file_path <- "..\\results\\SF\\d=10R=1.4\\test\\ctData\\viroRt.txt"
  SEIRData <- read.csv("..\\results\\SF\\d=10R=1.4\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 1.6)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=1.6\\test\\ctData\\ctDataID12.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=1.6\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\SF\\d=10R=1.6\\test\\ctData\\R0=1.6_univariate_chain.csv")
  file_path <- "..\\results\\SF\\d=10R=1.6\\test\\ctData\\viroRt.txt"
  SEIRData <- read.csv("..\\results\\SF\\d=10R=1.6\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 1.8)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=1.8\\test\\ctData\\ctDataID22.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=1.8\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\SF\\d=10R=1.8\\test\\ctData\\R0=1.8_univariate_chain.csv")
  file_path <- "..\\results\\SF\\d=10R=1.8\\test\\ctData\\viroRt.txt"
  SEIRData <- read.csv("..\\results\\SF\\d=10R=1.8\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 2.2)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=2.2\\test\\ctData\\ctDataID2.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=2.2\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\SF\\d=10R=2.2\\test\\ctData\\R0=2.2_univariate_chain.csv")
  file_path <- "..\\results\\SF\\d=10R=2.2\\test\\ctData\\viroRt.txt"
  SEIRData <- read.csv("..\\results\\SF\\d=10R=2.2\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 2.4)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=2.4\\test\\ctData\\ctDataID29.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=2.4\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\SF\\d=10R=2.4\\test\\ctData\\R0=2.4_univariate_chain.csv")
  file_path <- "..\\results\\SF\\d=10R=2.4\\test\\ctData\\viroRt.txt"
  SEIRData <- read.csv("..\\results\\SF\\d=10R=2.4\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 2.6)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=2.6\\test\\ctData\\ctDataID17.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=2.6\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\SF\\d=10R=2.6\\test\\ctData\\R0=2.6_univariate_chain.csv")
  file_path <- "..\\results\\SF\\d=10R=2.6\\test\\ctData\\viroRt.txt"
  SEIRData <- read.csv("..\\results\\SF\\d=10R=2.6\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 2.8)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=2.8\\test\\ctData\\ctDataID13.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=2.8\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\SF\\d=10R=2.8\\test\\ctData\\R0=2.8_univariate_chain.csv")
  file_path <- "..\\results\\SF\\d=10R=2.8\\test\\ctData\\viroRt.txt"
  SEIRData <- read.csv("..\\results\\SF\\d=10R=2.8\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 3.2)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=3.2\\test\\ctData\\ctDataID15.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=3.2\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\SF\\d=10R=3.2\\test\\ctData\\R0=3.2_univariate_chain.csv")
  file_path <- "..\\results\\SF\\d=10R=3.2\\test\\ctData\\viroRt.txt"
  SEIRData <- read.csv("..\\results\\SF\\d=10R=3.2\\test\\ctData\\SEIR_partab.csv")
}
if(net == "SF" & R0 == 3.4)
{
  ctData <- read.csv("..\\results\\SF\\d=10R=3.4\\test\\ctData\\ctDataID13.csv")
  RtData <- read.table("..\\results\\SF\\d=10R=3.4\\test\\ctData\\Rt.txt", header = FALSE)
  chain <- read.csv("..\\results\\SF\\d=10R=3.4\\test\\ctData\\R0=3.4_univariate_chain.csv")
  file_path <- "..\\results\\SF\\d=10R=3.4\\test\\ctData\\viroRt.txt"
  SEIRData <- read.csv("..\\results\\SF\\d=10R=3.4\\test\\ctData\\SEIR_partab.csv")
}

RtDatat <- t(RtData)
RtDatat <- as.data.frame(RtDatat)
num_rows <- nrow(RtDatat)
times_Rt <- 1:num_rows
times <- as.data.frame(times_Rt)

true_Rt <- cbind(times, RtDatat)

true_Rt_colnames <- c("t","prob_infection")
colnames(true_Rt) <- true_Rt_colnames

## MCMC chain options
mcmc_pars <- c("iterations"=15000,"popt"=0.44,"opt_freq"=1000,
               "thin"=50,"adaptive_period"=12000,"save_block"=1000)

chain <- chain[chain$sampno > mcmc_pars["adaptive_period"],]

getSEIR <- function(pars, times,N=100000){
  seir_pars <- c(pars["R0"]*(1/pars["infectious"]),1/pars["incubation"],1/pars["infectious"])
  init <- c((1-pars["I0"])*N,0,pars["I0"]*N,0,0)
  sol <- deSolve::ode(init, times, func="SEIR_model_lsoda",parms=seir_pars,
                      dllname="virosolver",initfunc="initmodSEIR",
                      nout=0, rtol=1e-10,atol=1e-10)
  use_colnames <- c("time","S","E","I","R","cumu_exposed")
  sol <- as.data.frame(sol)
  colnames(sol) <- use_colnames
  sol$Rt <- (sol$S) * pars["R0"] / N
  Rt <- sol$Rt
  Rt
}

predictions <- plot_prob_infection(chain,nsamps=200, INCIDENCE_FUNC=getSEIR,
                                   solve_times=times_Rt,obs_dat=ctData,
                                   true_prob_infection=true_Rt,smooth=TRUE)
p_incidence_prediction <- predictions$plot + scale_x_continuous(limits=c(0,num_rows))
p_incidence_prediction

result <- as.character(predictions$predictions$prob_infection[1])
for (i in 2:num_rows)
{
  result <- paste(result, as.character(predictions$predictions$prob_infection[i]))
}

result <- paste(result, "\n")
writeLines(result, file_path)
print(result)

