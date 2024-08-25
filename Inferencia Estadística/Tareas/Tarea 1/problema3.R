### Problema 3

# a)

library(ggplot2)

# DF


uniforme <- function(x, n) {
  if (x %in% c(1:n)) {
    return(1 / n)
  } else {
    return(0)
  }
}

df_graph <- function(n) {
  x <- c(1:n)
  y <- sapply(x, function(x_0) uniforme(x_0, n))

  data <- data.frame(x = x, y = y)

  ggplot(data = data, mapping = aes(x = x, y = y)) +
    geom_col(width = 0.05) +
    xlim(0, max(x) + 1) +
    ylim(0, 1.1 / n) +
    ylab("f(x)") +
    theme_classic()
  ggsave("C:/Users/gusta/Escritorio/p3incisoA1.tiff",
    width = 5, height = 4,
    units = "in", device = "tiff", dpi = 500
  )
}

df_graph(50) # Ejemplo con n = 50


# CDF




cdf_uniforme <- function(x, n) {
  # Regresa la función acumulativa de la
  # distribución U(x;n).
  prob <- 0
  if (x %in% c(1:n)) {
    return(NA)
  } else if (x < min(c(1:n))) {
    return(0)
  }
  for (i in c(1:n)) {
    if (x > i) {
      prob <- prob + uniforme(i, n)
    }
  }
  return(prob)
}


cdf_graph <- function(n, ruta) {
  # Guarda la gráfica de la función acumulativa
  # como F(x)_U(n)_graph.tiff en la ruta
  # especificada.
  x <- seq(-3, n + 3, by = 0.001)
  y <- sapply(x, function(x_0) cdf_uniforme(x_0, n))
  name <- paste0("F(x)_U(", n, ")_graph.tiff")

  ggplot(data.frame(x = x, y = y), aes(x, y)) +
    geom_line() +
    xlim(-3, max(x) + 1) +
    ylim(0, 1.1) +
    ylab("F(x)") +
    theme_classic()
  ggsave(paste(ruta, name),
    width = 5, height = 4,
    units = "in", device = "tiff", dpi = 500
  )
}

cdf_graph(30, "C:/Users/gusta/Escritorio/")




# c)
library(ggplot2)
n <- 10000

muestra <- sample(x = c(1:10), size = n, replace = TRUE, set.seed(13))

muestra_dataframe <- data.frame(muestra = muestra)

ggplot(data = muestra_dataframe) +
  geom_bar(mapping = aes(x = muestra), fill = "blue") +
  ylab("Frecuencia") +
  scale_x_discrete(
    name = "x",
    limits = c(1:10)
  ) +
  ylim(0, 0.11 * n) +
  theme_classic()
ggsave("C:/Users/gusta/Escritorio/test.png", )
