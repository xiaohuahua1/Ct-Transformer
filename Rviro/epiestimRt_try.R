R0 <- 3.4
net <- "SF"

if(net == "ER" & R0 == 1.4)
{
  dailyNum <- read.csv("..\\results\\ER\\d=10R=1.4\\test\\ctData\\dailyNumID9.csv")
  Generation <- read.csv("..\\results\\ER\\d=10R=1.4\\test\\ctData\\GenerationID9.csv")
}
if(net == "ER" & R0 == 1.8)
{
  dailyNum <- read.csv("..\\results\\ER\\d=10R=1.8\\test\\ctData\\dailyNumID14.csv")
  Generation <- read.csv("..\\results\\ER\\d=10R=1.8\\test\\ctData\\GenerationID14.csv")
}
if(net == "ER" & R0 == 2.2)
{
  dailyNum <- read.csv("..\\results\\ER\\d=10R=2.2\\test\\ctData\\dailyNumID5.csv")
  Generation <- read.csv("..\\results\\ER\\d=10R=2.2\\test\\ctData\\GenerationID5.csv")
}
if(net == "ER" & R0 == 2.4)
{
  dailyNum <- read.csv("..\\results\\ER\\d=10R=2.4\\test\\ctData\\dailyNumID15.csv")
  Generation <- read.csv("..\\results\\ER\\d=10R=2.4\\test\\ctData\\GenerationID15.csv")
}
if(net == "ER" & R0 == 2.6)
{
  dailyNum <- read.csv("..\\results\\ER\\d=10R=2.6\\test\\ctData\\dailyNumID20.csv")
  Generation <- read.csv("..\\results\\ER\\d=10R=2.6\\test\\ctData\\GenerationID20.csv")
  
}
if(net == "ER" & R0 == 2.8)
{
  dailyNum <- read.csv("..\\results\\ER\\d=10R=2.8\\test\\ctData\\dailyNumID11.csv")
  Generation <- read.csv("..\\results\\ER\\d=10R=2.8\\test\\ctData\\GenerationID11.csv")
}
if(net == "ER" & R0 == 3.2)
{
  dailyNum <- read.csv("..\\results\\ER\\d=10R=3.2\\test\\ctData\\dailyNumID18.csv")
  Generation <- read.csv("..\\results\\ER\\d=10R=3.2\\test\\ctData\\GenerationID18.csv")
}
if(net == "ER" & R0 == 3.4)
{
  dailyNum <- read.csv("..\\results\\ER\\d=10R=3.4\\test\\ctData\\dailyNumID14.csv")
  Generation <- read.csv("..\\results\\ER\\d=10R=3.4\\test\\ctData\\GenerationID14.csv")
}
if(net == "SF" & R0 == 1.2)
{
  dailyNum <- read.csv("..\\results\\SF\\d=10R=1.2\\test\\ctData\\dailyNumID30.csv")
  Generation <- read.csv("..\\results\\SF\\d=10R=1.2\\test\\ctData\\GenerationID30.csv")
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
  dailyNum <- read.csv("..\\results\\SF\\d=10R=2.4\\test\\ctData\\dailyNumID29.csv")
  Generation <- read.csv("..\\results\\SF\\d=10R=2.4\\test\\ctData\\GenerationID29.csv")
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
  dailyNum <- read.csv("..\\results\\SF\\d=10R=3.4\\test\\ctData\\dailyNumID7.csv")
  Generation <- read.csv("..\\results\\SF\\d=10R=3.4\\test\\ctData\\GenerationID7.csv")
}


MCMC_seed <- 1
overall_seed <- 2

R_si_from_sample <- estimate_R(incid=dailyNum,
                               method = "si_from_data",
                               si_data = Generation,
                               config = make_config(list(si_parametric_distr = "L",
                                            n1 = 20, n2 = 10,
                                            seed = overall_seed)))
plot(R_si_from_sample, legend=FALSE)

total <- nrow(dailyNum)
real_total <- nrow(R_si_from_sample$R)
result <- as.character(R_si_from_sample$R$`Mean(R)`[1])

for (i in 2:real_total)
{
  result <- paste(result, as.character(R_si_from_sample$R$`Mean(R)`[i]))
}

for (i in 1:(total - real_total))
{
  result <- paste(result, as.character(0))
}
result <- paste(result, "\n")
result

file_path <- "EpiRt.txt"
writeLines(result, file_path)
