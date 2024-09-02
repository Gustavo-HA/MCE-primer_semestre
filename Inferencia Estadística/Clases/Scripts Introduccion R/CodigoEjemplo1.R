## Operaciones aritmeticas

## Exponenciacion
8^(1/3)

## Absoluto
abs(5.7-6.8)/0.38

## Divisor y modulo (residuo)
119 %/% 13
119 %% 13

## Redondeo
floor(5.7)
ceiling(5.7)

## Funciones redondeos
rounded <- function(x) floor(x+0.5)
rounded = function(x) floor(x+0.5)

rounded <- function(x) {
  floor(x+0.5)
}

rounded(5.1)

## Secuencias
1:7

seq(0,1,0.1)

seq(5,-5,-1)

seq(from=5,by= -1, along=1:20)

seq(by= -1, from=5, along=1:20)


a <- seq(1,10^6,by=100)
a = seq(1,10^6,by=100)

## Repeticiones
rep(5.3,17)
rep(1:6,2)

## Niveles
rep(1:6,rep(3,6))

debug(rep)

debug(gl)
gl(6,3,18)
undebug(gl)


rep(1:6,1:6)

rep(1:6,c(1,2,3,3,2,1))

rep(c("monoecious","dioecious","hermaphrodite","agamic"),c(3,2,7,3))

## Vectorizacion
y <- c(7,5,7,2,4,6,1,6,2,3)

y <- c(y,y)

sum((y-mean(y))^2)/(length(y)-1)

var(y)

## Ordenamiento

data <- c(7,4,6,8,9,1,0,3,2,5,0)

ranks <- rank(data)
sorted <- sort(data)
ordered <- order(data)

data[ordered]

view <- data.frame(data,ranks,sorted,ordered)
view

## Uso de dataframes
worms <- read.csv("worms.csv",header=T,row.names=1)
attach(worms)
names(worms)
summary(worms)
str(worms)

## Subseleccion
worms[,1:3]

worms[5:15,]

worms[5:15,3:4]

worms[Area>3 & Slope <3 , ]

worms[2:11,][worms[1:10,][,"Slope"]< 3 & worms[1:10,][,"Area"] > 3,]



             ## Ordenamiento de elementos

worms[order(worms[,1]),c(1:6)]
worms[order(worms[,1]), c("Area", "Slope", "Vegetation", "Soil.pH",  "Damp", "Worm.density")]

worms[rev(order(worms[,4])),c(4,6)]

## Uso de vectores y funciones
x <- 0:10
sum(x)

sum(x<5)

sum(x[x<5])

## Suma de los tres valores m?s grandes
y <- c(8,3,5,7,6,6,8,9,2,3,9,4,10,4,11)
sort(y)
rev(sort(y))
sum(rev(sort(y))[1:3])

## Indices
y <- c( 8, 3, 5, 7, 6, 6, 8, 9, 2, 3, 9, 4, 10, 4, 11 )
idx <- which(y>5)

y[y>5]
y[idx]

## Muestreos
sample(y)
sample(y,replace=T)


## Aritmetica logica
ys <- y[y<5]
ys

yb <- y[y>=5]
yb


## Localizacion de elementos
## Cuantos
sum(y > mean(y)+2 | y < mean(y)-2 )

## Suma
sum(y [y> mean(y)+2 | y < mean(y)-2] )


## Reemplazo de elementos
y <- c ( -8, 3, 5, 7, 6, 6, 8, 9, 2, 3, 9, 4, 10, 4, 11 )

for (i in 1:length(y)) { 
  if(y[i] < 0 ) 
    y[i] <- 0 
}

y[ y< 0 ] <- 0

y <- ifelse( y < 0 , -1, 1 )

## Indices negativos
x<- c(5,8,6,7,1,5,3)
y <- x[-1]
y <- x[-c(1,2)]
y

trim.mean <- function (x) mean(sort(x)[-c ( 1, length(x) ) ])

trim.mean(x)

## Grafica de funciones
x <- 0:50
y <- 3*(1-exp(-0.1*x))
plot(x,y,type="l")

x <- seq(-5,5,.1)
y <- exp(.1+.4*x)/(1+exp(.1+.4*x))
plot(x,y,type="l",ylim=c(0,1))

## Otra funcion
y2 <- exp(.1+.6*x)/(1+exp(.1+.6*x))
lines(x,y2,lty=2)

y3 <- exp(.1+1.6*x)/(1+exp(.1+1.6*x))
lines(x,y3,lty=4)

## Matrices
y <- c( 8, 3, 5, 7, 6, 6, 8, 9, 2, 3, 9, 4, 10, 4, 11)
m <- matrix(y,nrow=5)
m

## Aritmetica matricial
## Matriz de fecundidades y sobrevivencia
L <- c(0,0.7,0,0,6,0,0.5,0,3,0,0,0.3,1,0,0,0)
L <- matrix(L,nrow=4)

E <- matrix(nrow=8,ncol=6,1)

n <- c(45,20,17,3)
n <- matrix(n,ncol=1)
n

L %*% n

## Solucion de sistemas de ecuaciones lineales
## 3x + 4y = 12
## x + 2y=8
A <- matrix(c(3,1,4,2),nrow=2)

kv <- matrix(c(12,8),nrow=2)
kv

solve(A,kv)

## Uso para el calculo de estadisticas
fxp <- rnorm(33,15,4)
fyp <- rchisq(33, 7)

distribution <- data.frame(fx=fxp, fy=fyp)
attach(distribution)
names(distribution)

## Mediana
par(mfrow=c(1,2))
hist(fx)
hist(fy)

## Calcula la mediana
y <- sort(fy)
y2 <- sort(fx)
length(y)
ceiling(length(y)/2)

y[17]
y[ceiling(length(y)/2)]
median(y)

## Media
sum(y)/length(y)
mean(y)

sum(y2)/length(y2)
mean(y2)

## Media geometrica
prod(y)^(1/length(y))

meanlogy<-sum(log(y))/length(y)
meanlogy

exp(meanlogy)

geometric <- function(x) exp(sum(log(x))/length(x))
geometric(y)

## Ejemplo de media geometrica
aphid <- c(10,1,1,10,1000)
mean(aphid)
geometric(aphid)

## Ejemplo de media armonica
mean(c(1,2,4,1))

4/sum(1/c(1,2,4,1))
harmonic<-function(x) 1/mean(1/x)

harmonic(y)

## Medidas de variacion
rm(list=ls())
par(mfrow=c(1,1))
y <- c(13,7,5,12,9,15,6,11,9,7,12)
plot(y,ylim=c(0,20))

d = range(y)
abline(mean(y),0)
abline(d[1],0, col="red")
abline(d[2],0, col="red")

## Ejemplo de varianza
A<-c(3,4,4,3,2,3,1,3,5,2)
B<-c(5,5,6,7,4,4,3,5,6,5)
C<-c(3,3,2,1,10,4,3,11,3,10)

mean(A)
mean(B)
mean(C)

dA <- A-3
dB <- B-5
dC <- C-5

SSA<-sum(dA^2)
SSB<-sum(dB^2)
SSC<-sum(dC^2)

s2A<-SSA/9
s2B<-SSB/9
s2C<-SSC/9

s2A;s2B;s2C

s2A<-var(A)

## errore3s estandar
sqrt(s2A/10)
sqrt(s2B/10)
sqrt(s2C/10)

## Intervalos de confianza
qt(.025,9)
qt(.975,9)

qt(.995,9)
qt(.9975,9)

qt(.975,9)*sqrt(1.33333/10)

## Cuantiles
z <- rnorm(1000)
mean(z)
quantile(z,c(.025,0.5,.975))

z<-rnorm(10000)
quantile(z,c(.025,.975))

## Medidas alter de varianza
y<- c(3,4,6,4,5,2,4,5,1,5,4,6)
mad(y)
sd(y)

y1<-c(y,100)
mean(y1)
sqrt(var(y1))

mad(y1)

outlier<-function(x) {
  if(sqrt(var(x))>4*mad(x)) print("Hay outliers")
  else print("Desviaciones razonables") }

outlier(y1)

## Estimaciones de la media en muestras
light<-read.csv("d:/Edgar/CursoEstadistica/Sesion3/light.csv",header=T)
attach(light)
names(light)

hist(speed)

summary(speed)
library(stats)
wilcox.test(speed,mu=990)

t.test(speed,mu=990)

## Comparaciones de dos medias
A
B
C

qt(.975,18)
(mean(A)-mean(B))/sqrt(s2A/10+s2B/10)

t.test(A,B)

## test no parametricos
par(mfrow=c(1,2))
hist(B,breaks=c(0.5:11.5))
hist(C,breaks=c(0.5:11.5))

combined<-c(B,C)
combined

sample<-c(rep("B",10),rep("C",10))
sample

rank.combi<-rank(combined)
rank.combi

sum(rank.combi[sample=="B"])
sum(rank.combi[sample=="C"])

tapply(rank.combi,sample,sum)

test = wilcox.test(B,C)

## Ejemplo de muestras dependientes
x<-c(20,15,10,5,20,15,10,5,20,15,10,5,20,15,10,5)
y<-c(23,16,10,4,22,15,12,7,21,16,11,5,22,14,10,6)

t.test(x,y)

t.test(x,y,paired=T)


## Teorema del l?mite central
par(mfrow=c(1,2))
y<-rnbinom(1000,1,.2)

mean(y)
var(y)
table(y)
hist(y)

my <- numeric(1000)
for (i in 1:1000) {
  y <- rnbinom(30, 1, 0.2)
  my[i] <- mean(y) 
  }
hist(my)

## Regresion
par(mfrow=c(1,1))
regression<-read.csv("d:/Edgar/CursoEstadistica/Sesion3/regression.csv",header=T)
attach(regression)
names(regression)
plot(tannin,growth)
mean(growth)
abline(6.889,0)

sum(tannin);sum(tannin^2)
sum(growth);sum(growth^2)
sum(tannin*growth)

qf(0.95,1,7)
plot(tannin,growth)
abline(lm(growth~tannin))

model <- lm(growth~tannin)

summary(model)

par(mfrow=c(2,2))
plot(model)

predict(model,list(tannin=5.5))

par(mfrow=c(1,1))
rm(x,y)

## Segundo ejemplo
decay<-read.csv("d:/Edgar/CursoEstadistica/Sesion3/decay.csv",header=T)
attach(decay)
names(decay)
plot(time,amount)
abline(lm(amount~time))

result<-lm(amount~time)
summary(result)
plot(result)


plot(time,amount,log="y")

transformed <- lm(log(amount)~time)
summary(transformed)

plot(transformed)


plot(time,amount)
smoothx <- data.frame(time=seq(0,30,0.1))
smoothy <-exp(predict(transformed,smoothx))
lines(as.numeric(smoothx$time),as.numeric(smoothy) )
