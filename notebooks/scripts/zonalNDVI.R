library(terra)
library(sf)
library(ggplot2)

# Import dataset
det <- rast('../../predictions/det_merged.tif')
ndvi <- rast('../../rawImage/ndvi_merged.tif')
rgi <- rast('../../rawImage/rgi_merged.tif')
image <- rast('../../source/out.tif')
ann <- read_sf('../../training/trainingPolygon.shp')

# Zonal statistics of manually delineated tree crowns
ann$label[is.na(ann$label)] <- 2
ann$meanNDVI <- extract(ndvi, ann, fun=mean, na.rm=TRUE)
ann$meanRGI <- extract(rgi, ann, fun=mean, na.rm=TRUE)

ndviDead <- mask(ndvi, det, maskvalues=c(0,2), updatevalue=NA)
ndviLive <- mask(ndvi, det, maskvalues=c(0,1), updatevalue=NA)

r <- rast(nrows=5, ncols=5)
values(r) <- -10:14

x <- ifel(r > 1, 1, r)

deadmask <- 

  
ggplot(as.data.frame(ndviDead))

###MAKE FIGURE####
# Plot NDVI
ndvi_p <- ggplot(ann, aes(x=meanNDVI, y=..density.., group=label, color=factor(label), fill=factor(label))) +
  geom_histogram(alpha=0.5, bins=40, position="identity") +
  labs(title="Histogram Plot", x = "NDVI", y="Probability Density")+
  scale_color_manual(values=c("#fc8d59", "#91cf60")) +
  scale_fill_manual(values=c("#fc8d59", "#91cf60"),
                    name='Status',
                    labels=c('Dead', 'Live')) +
  guides(colour = "none") +
  theme(plot.title = element_text(hjust = 0.5), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_blank(), panel.border = element_blank(),
        axis.line.y = element_line(colour = 'black', size=0.3, linetype='solid'),
        axis.line.x = element_line(colour = 'black', size=0.3, linetype='solid'), 
        axis.text.y = element_text(hjust = 0))
ndvi_p


# Plot RGI
rgi_p <- ggplot(ann, aes(x=meanRGI, y=..density.., group=label, color=factor(label), fill=factor(label))) +
  geom_histogram(alpha=0.5, bins=40, position="identity") +
  labs(title="Histogram Plot", x = "NDVI", y="Probability Density")+
  scale_color_manual(values=c("#fc8d59", "#91cf60")) +
  scale_fill_manual(values=c("#fc8d59", "#91cf60"),
                    name='Status',
                    labels=c('Dead', 'Live')) +
  guides(colour = "none") +
  theme(plot.title = element_text(hjust = 0.5), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_blank(), panel.border = element_blank(),
        axis.line.y = element_line(colour = 'black', size=0.3, linetype='solid'),
        axis.line.x = element_line(colour = 'black', size=0.3, linetype='solid'), 
        axis.text.y = element_text(hjust = 0))
rgi_p
