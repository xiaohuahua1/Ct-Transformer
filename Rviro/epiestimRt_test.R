net <- "SF"
type <- 4

if(net == "ER" & type == 0)
{
  dailyNum <- read.csv("..\\results\\ER\\testRate\\ER2.4\\ctData\\dailyNumID9.csv")
  Generation <- read.csv("..\\results\\ER\\testRate\\ER2.4\\ctData\\GenerationID9.csv")
  file_path <- "..\\results\\ER\\testRate\\ER2.4\\draw\\Epi_all.txt"
}
if(net == "ER" & type == 1)
{
  dailyNum <- read.csv("..\\results\\ER\\testRate\\ER2.4\\ctData\\dailyNumID1-9.csv")
  Generation <- read.csv("..\\results\\ER\\testRate\\ER2.4\\ctData\\GenerationID1-9.csv")
  file_path <- "..\\results\\ER\\testRate\\ER2.4\\draw\\Epi_test1.txt"
}
if(net == "ER" & type == 2)
{
  dailyNum <- read.csv("..\\results\\ER\\testRate\\ER2.4\\ctData\\dailyNumID2-9.csv")
  Generation <- read.csv("..\\results\\ER\\testRate\\ER2.4\\ctData\\GenerationID2-9.csv")
  file_path <- "..\\results\\ER\\testRate\\ER2.4\\draw\\Epi_test2.txt"
}
if(net == "ER" & type == 3)
{
  dailyNum <- read.csv("..\\results\\ER\\testRate\\ER2.4\\ctData\\dailyNumID3-9.csv")
  Generation <- read.csv("..\\results\\ER\\testRate\\ER2.4\\ctData\\GenerationID3-9.csv")
  file_path <- "..\\results\\ER\\testRate\\ER2.4\\draw\\Epi_test3.txt"
}
if(net == "ER" & type == 4)
{
  dailyNum <- read.csv("..\\results\\ER\\testRate\\ER2.4\\ctData\\dailyNumID4-9.csv")
  Generation <- read.csv("..\\results\\ER\\testRate\\ER2.4\\ctData\\GenerationID4-9.csv")
  file_path <- "..\\results\\ER\\testRate\\ER2.4\\draw\\Epi_test4.txt"
}
if(net == "SF" & type == 0)
{
  dailyNum <- read.csv("..\\results\\ER\\testRate\\SF2.4\\ctData\\dailyNumID7.csv")
  Generation <- read.csv("..\\results\\ER\\testRate\\SF2.4\\ctData\\GenerationID7.csv")
  file_path <- "..\\results\\ER\\testRate\\SF2.4\\draw\\Epi_all.txt"
}
if(net == "SF" & type == 1)
{
  dailyNum <- read.csv("..\\results\\ER\\testRate\\SF2.4\\ctData\\dailyNumID1-7.csv")
  Generation <- read.csv("..\\results\\ER\\testRate\\SF2.4\\ctData\\GenerationID1-7.csv")
  file_path <- "..\\results\\ER\\testRate\\SF2.4\\draw\\Epi_test1.txt"
}
if(net == "SF" & type == 2)
{
  dailyNum <- read.csv("..\\results\\ER\\testRate\\SF2.4\\ctData\\dailyNumID2-7.csv")
  Generation <- read.csv("..\\results\\ER\\testRate\\SF2.4\\ctData\\GenerationID2-7.csv")
  file_path <- "..\\results\\ER\\testRate\\SF2.4\\draw\\Epi_test2.txt"
}
if(net == "SF" & type == 3)
{
  dailyNum <- read.csv("..\\results\\ER\\testRate\\SF2.4\\ctData\\dailyNumID3-7.csv")
  Generation <- read.csv("..\\results\\ER\\testRate\\SF2.4\\ctData\\GenerationID3-7.csv")
  file_path <- "..\\results\\ER\\testRate\\SF2.4\\draw\\Epi_test3.txt"
}
if(net == "SF" & type == 4)
{
  dailyNum <- read.csv("..\\results\\ER\\testRate\\SF2.4\\ctData\\dailyNumID4-7.csv")
  Generation <- read.csv("..\\results\\ER\\testRate\\SF2.4\\ctData\\GenerationID4-7.csv")
  file_path <- "..\\results\\ER\\testRate\\SF2.4\\draw\\Epi_test4.txt"
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
writeLines(result, file_path)

