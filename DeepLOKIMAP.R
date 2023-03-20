# https://mikkovihtakari.github.io/PlotSvalbard/articles/PlotSvalbard.html
library(PlotSvalbard)
library(devtools)
# devtools::install_github("MikkoVihtakari/PlotSvalbard", upgrade = "never")
#basemap("barentssea", limits = c(-6, 20, 75, 83), bathymetry = TRUE, bathy.style = "contour_grey")
#basemap("barentssea", bathymetry = TRUE, bathy.style = "poly_greys", currents = TRUE, current.alpha = 0.7)
data("npi_stations")
print(npi_stations)
locs  <- npi_stations[npi_stations$Station %in% c("Kb4"), ]
print(locs)
sb = list("Framstrait", "R5", 15.01, 79.01) #check coordinates
gl = list("Framstrait", "R6", -13.0, 81.51) #check coordinates
hgb = list("Framstrait", "HG4", 3.8, 79.31) #check coordinates
fb = list("Framstrait", "KH", 7.1, 79.30) #check coordinates
#
zero = list("Framstrait", "R1", -0.4007, 79.262) #check coordinates
egc3 =list("Framstrait", "R2", -3.934, 79.151) #check coordinates
n5 =list("Framstrait", "R3", 3.326, 80.250) #check coordinates
s3 =list("Framstrait", "R4", 5.21, 78.307) #check coordinates

# 
locs <- rbind(locs, sb, gl, hgb, fb, zero, egc3, n5, s3)
print(locs)
x <- transform_coord(locs, lon = "Lon", lat = "Lat", bind = TRUE)
x$Label <- c("Dummy", "Svalbard", "Greenland", "H4", "F4", "Z","E3", "N5", "S3")
y <- x[x$Station %in% c("R5","R6"), ]
f <- x[x$Station %in% c("HG4","KH","R1","R2","R3","R4"), ]

pointdata <- data.frame(
  Lon = c(4.18, 7, -0.0007, -3.934, 3.026, 4.969), 
  Lat = c(79, 79, 78.962 , 78.851, 79.990, 78.607), 
  Label = c('HG4', 'F4', "Z","E3", "N5", "S3"),
  moorings = c('HG4','F4', "Z","E3", "N5", "S3"),
  ptyname = c("p", "q", "Z","E3", "N5", "S3")
) 

pd  <- transform_coord(pointdata, lon = "Lon", lat = "Lat", bind = TRUE)

pd1 <- pd[pd$Label %in% c("HG4"), ]
pd2 <- pd[pd$Label %in% c("HG4","F4","Z","E3", "N5", "S3"), ]

pdf("last_try_framstrait_map_dots_currents.pdf")
basemap("barentssea",  bathymetry = TRUE, bathy.style = "poly_blues", limits = c(-5, 20, 75, 83), round.lat = 2, round.lon = 4) + geom_text(data = f, aes(x = lon.utm, y = lat.utm, label = Label), color = "indianred3", fontface = 10, size =6.5)+ geom_text(data = y, aes(x = lon.utm, y = lat.utm, label = Label), color = "black", fontface = 10, size=7.5)  + geom_point(data = pd2,, color ="indianred3",  mapping = aes(x = lon.utm, y = lat.utm),size=3)
dev.off()

basemap("barentssea",  bathymetry = TRUE, bathy.style = "poly_blues", limits = c(-5, 20, 75, 83), round.lat = 2, round.lon = 4) + geom_text(data = f, aes(x = lon.utm, y = lat.utm, label = Label), color = "indianred3", fontface = 10, size =6.5)+ geom_text(data = y, aes(x = lon.utm, y = lat.utm, label = Label), color = "black", fontface = 10, size=7.5)  + geom_point(data = pd2,, color ="indianred3",  mapping = aes(x = lon.utm, y = lat.utm),size=3)

ggsave("FramStraitMap_currents.jpeg", device = "jpeg", dpi=300)
#basemap("barentssea", bathymetry = TRUE, bathy.style = "poly_blues", limits = c(-5, 20, 75, 83)) + geom_point(data = pd, mapping = aes(x = lon.utm, y = lat.utm, label = Label, color =color))
