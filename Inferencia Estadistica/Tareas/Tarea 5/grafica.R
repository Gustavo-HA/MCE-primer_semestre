library(ggplot2)
library(httpgd)
library(extrafont)
font_import(pattern = "lmroman*")
library(latex2exp)


# Crear un data frame con los puntos del triángulo
triangle_data <- data.frame(
  x = c(0, 1, 0), # Coordenadas x
  y = c(0, 0, 1) # Coordenadas y
)



# Generar la gráfica
ggplot() +
  geom_polygon(
    data = triangle_data, aes(x = x, y = y),
    fill = "gray", color = "black"
  ) + # Área sombreada y bordes
  geom_segment(aes(x = -0.2, xend = 1.2, y = 0, yend = 0)) +
  geom_segment(aes(x = 0, xend = 0, y = -0.2, yend = 1.2)) +
  annotate("text",
    x = 0, y = 0, label = "(0,0)",
    size = 5, color = "black", family = "LM Roman 10"
  ) +
  annotate("text",
    x = 1, y = 0, label = "(1,0)",
    size = 5, color = "black", family = "LM Roman 10"
  ) +
  annotate("text",
    x = 0, y = 1, label = "(0,1)",
    size = 5, color = "black", family = "LM Roman 10"
  ) +
  scale_x_continuous(breaks = c(0, 1)) + # Escala de x
  scale_y_continuous(breaks = c(0, 1)) + # Escala de y
  labs(x = NULL, y = NULL) + # Etiquetas de ejes
  theme_minimal() + # Tema limpio
  theme(
    panel.grid = element_blank(), # Sin líneas de cuadrícula
    axis.line = element_blank(), # Líneas de los ejes
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank(),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    text = element_text(family = "LM Roman 10")
  )
