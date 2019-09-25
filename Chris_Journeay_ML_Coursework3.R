################## Coursework 3 ##################################
# Chris Journeay 
# ML-Coursework 3


########### 1 - Environment Initialisation ########################

#sets working directory
setwd("C:/Users/chris/Desktop/Machine Learning/Labs/Coursework 3")
#clears the memory 
rm(list = ls())
#loads libraries
library(factoextra)
library(cluster)
library(tidyverse)
library(dendextend)
library(stats)
library(FactoMineR)
library(Rfast)
library(corrplot)
library(reshape2)
# sets random seed
set.seed(713)
# imports the data
whiskies <- read.csv("whisky.csv", header=TRUE, sep = ",", row.names = "Distillery")
#retains the full malt names as a separate table
malt_names <- whiskies["Malt.Name"]
#drops non flavour profile columns
whiskies <- whiskies[,2:13]

################## 2 - Data Exploration ################################

#creates density plot
melt.whiskies <- melt(whiskies)

ggplot(data = melt.whiskies, aes(x = value)) + 
  stat_density() + 
  facet_wrap(~variable, scales = "free")

#creates correlation plot
wcor <- cor(whiskies)
corrplot(wcor, method = "square")

#gets principle components
principle_components <- PCA(whiskies, scale. = FALSE)

#Eigenvalues
pc_eigen <- get_eigenvalue(principle_components)

pc_eigen

fviz_eig(principle_components, 
         palette = "aaas",
         addlabels = TRUE, 
         repel = TRUE, 
         ylim = c(0, 35),
         main = "Scree Plot of Eigenvalues")


#plots individuals
fviz_pca_ind(principle_components,
             repel = TRUE,
             lab_size = 6,
             col.ind = "dark green")

#plots variables against principle components
fviz_pca_var(principle_components,
             col.var = "red")

#biplot of invidividuals 
fviz_pca(principle_components,
         lab_size = 6, 
         repel = TRUE,
         col.ind = "dark grey",
         col.var = "red")


fviz_cos2(principle_components, choice = "var", axes = 1:2)

# Contributions of variables to PC1
fviz_contrib(principle_components, choice = "var", axes = 1)
# Contributions of variables to PC2
fviz_contrib(principle_components, choice = "var", axes = 2)

# derives Hopskins statistic
hop <- get_clust_tendency(whiskies, n=nrow(whiskies)-1)
hop$hopkins_stat

#gets dissimiliarity matrix using the spearman distance
diss_spearman = get_dist(whiskies, method = "spearman", stand = FALSE)
# Visualize the dissimilarity matrix
fviz_dist(diss_spearman, order = TRUE, show_labels = TRUE, lab_size = 7,
          gradient = list(low = "blue", mid = "white", high = "red"))
          
#gets dissimiliarity matrix using the kendall distance
diss_kendall = get_dist(whiskies, method = "kendall", stand = FALSE)
# Visualize the dissimilarity matrix
fviz_dist(diss_kendall, order = TRUE, show_labels = TRUE, lab_size = 7,
          gradient = list(low = "blue", mid = "white", high = "red"))


###################### 3 - k-means clustering ##################################################

#determine optimal clusters
fviz_nbclust(whiskies, kmeans, method = "silhouette")

chosen_k = 5

#Compute k-means
k_means <- kmeans(diss_kendall, chosen_k, nstart = 25)

# Visualize clusters using factoextra
fviz_cluster(k_means, whiskies,
             ellipse.type = "norm", 
             stand = FALSE, 
             repel = TRUE,
             labelsize = 8,
             show.clust.cent = TRUE,
             ellipse.alpha = 0.25,
             palette = "aaas", 
             ggtheme = theme_light(),
             main = "K-means Clustering with k = 5",
             legend = "right")

#gets silhoutte values and plots them
sil <- silhouette(k_means$cluster, dist(whiskies))
fviz_silhouette(sil, 
                palette = "aaas", 
                ggtheme = theme_light(),
                main = "K-means Clustering with k = 5")

########################## 4 - k-mediod clustering  #######################################################

#determine optimal clusters
fviz_nbclust(whiskies, pam, method = "silhouette")

chosen_k = 3

k_mediods <- pam(diss_kendall, chosen_k, diss = TRUE, stand = FALSE )
k_mediods$data <- whiskies
print(k_mediods$clusinfo)

fviz_cluster(k_mediods,
             repel = TRUE,    
             ellipse.type = "norm",
             ellipse.alpha = 0.25,
             show.clust.cent = TRUE, 
             stand = FALSE, 
             labelsize = 8,
             palette = "aaas",        
             ggtheme = theme_light(),
             main = "K-mediod clustering with k = 3",
             legend = "right")

sil <- silhouette(k_mediods$cluster, dist(whiskies))
fviz_silhouette(sil, 
                palette = "aaas", 
                ggtheme = theme_light(),
                main = "K-mediod clustering with k = 3")

#gets counts of misclassified observations
sum(sil[,3] < 0)

##################### 5 - Aglomerative Hierarchical Clustering ##########################################

#Agglomerative Hierarchical Clustering

# methods to assess
ahc_meth <- c( "average", "single", "complete", "ward")
names(ahc_meth) <- c( "average", "single", "complete", "ward")

# function to compute coefficient
ac <- function(x) {
  agnes(diss_kendall, diss = TRUE, method = x)$ac
}

map_dbl(ahc_meth, ac)

#runs the ahglomerative clustering using the chosen method

ahc <- agnes(diss_kendall, diss = TRUE, stand = FALSE,   method = "ward")
pltree(ahc, cex = 0.6, hang = -1, main = "Dendrogram of Agglomerative Hierarchical Clustering")
abline(h=1.6, col = "red")

chosen_k = 3

# Cut tree into groups
ahc_sub_grp <- cutree(as.hclust(ahc), chosen_k)
fviz_cluster(list(data = whiskies, cluster = ahc_sub_grp),
             ellipse.type = "norm",
             ellipse.alpha = 0.25,
             show.clust.cent = TRUE, 
             stand = FALSE, 
             labelsize = 8,
             palette = "aaas",
             main = "Agglomerative Hierarchical Clusters with k = 3")

fviz_dend(ahc, 
          chosen_k,
          cex = 0.7,                    
          palette = "aaas",              
          rect = TRUE, rect_fill = TRUE, 
          rect_border = "aaas",           
          labels_track_height = 0.8,      
          ggtheme = theme_light(),
          main = "Agglomerative Hierarchical Clustering Dendrogram with k = 3",
          legend = "right")

sil <- silhouette(ahc_sub_grp, dist(whiskies))
rownames(sil) <- rownames(whiskies)
fviz_silhouette(sil, 
                palette = "aaas", 
                ggtheme = theme_light(),
                main = "Agglomerative Hierarchical clustering with k = 3")

#gets counts of misclassified observations
sum(sil[,3] < 0)

########################### 6 - Divisive Hierarchical Clustering ############################# 

# compute divisive hierarchical clustering
dhc <- diana(diss_kendall, diss=TRUE, stand = FALSE)

# Divise coefficient; amount of clustering structure found
dhc$dc

# plot dendrogram
pltree(dhc, cex = 0.6, hang = -1, main = "Dendrogram of Divisive Hierarchical Clustering")
abline(h=1.0, col = "red")

chosen_k = 5

# Cut tree into 5 groups
dhc_sub_grp <- cutree(as.hclust(dhc), chosen_k)
fviz_cluster(list(data = whiskies, cluster = dhc_sub_grp),
             palette = "aaas",
             ellipse.type = "norm",
             ellipse.alpha = 0.25,
             show.clust.cent = TRUE, 
             stand = FALSE, 
             labelsize = 7,
             main = "Divisive Hierarchical Clusters with k=5")


fviz_dend(dhc, 
          chosen_k,
          cex = 0.6,                     
          palette = "aaas",              
          rect = TRUE, rect_fill = TRUE,
          rect_border = "aaas",         
          labels_track_height = 0.8,    
          ggtheme = theme_light(),
          main = "Divisive Hierarchical Clustering Dendrogram with k = 5",
          legend = "right")

sil <- silhouette(dhc_sub_grp, dist(whiskies))
rownames(sil) <- rownames(whiskies)
print(sil[, 1:3])

fviz_silhouette(sil, 
                palette = "aaas", 
                ggtheme = theme_light(),
                main = "Divisive Hierarchical clustering with k = 5")

#gets counts of misclassified observations
sum(sil[,3] < 0)

########################### 7 - Fuzzy Clustering ############################################

chosen_k = 6

fuzzy <- fanny(diss_kendall, chosen_k , diss=TRUE, memb.exp = 1.2, stand = FALSE) 
fuzzy$data <- whiskies
head(fuzzy$membership,20) # Membership coefficients
fuzzy$coeff

fviz_cluster(fuzzy, 
             labelsize = 9,
             ellipse.type = "norm", 
             ellipse.alpha = 0.25,
             stand = FALSE, 
             repel = TRUE,
             show.clust.cent = TRUE,
             palette = "aaas", 
             ggtheme = theme_light(),
             main = "Fuzzy Clustering with k = 6",
             legend = "right")
fviz_silhouette(fuzzy, 
                palette = "aaas", 
                ggtheme = theme_light(),
                main = "Fuzzy clustering with k = 6")

#returns number of misclassified observations
sum(fuzzy$silinfo$widths < 0)

######################## 8 - Chosen Model ##############################################


chosen_k = 6

fuzzy <- fanny(diss_kendall, chosen_k , diss=TRUE, memb.exp = 1.2, stand = FALSE) 

#adds subgroups to original dataset
whiskies$cluster <- fuzzy$clustering

for (i in 1:chosen_k){
  print(head(subset(whiskies, whiskies$cluster == i), 30))
}

#adds cluster-ID to distance matrix
dist_matrix <- as.matrix(diss_kendall)
dist_matrix <- data.frame(cbind(fuzzy$clustering, dist_matrix))
names(dist_matrix)[names(dist_matrix) == "V1"] <- "cluster"

#function to return the nth recommended whisky name - returns nth closest distance from the row
# as the shortest distance is always zero (distance to itself) we add one to the number to ensure we return the correct one 
get_recc <- function(x, row, df, lim){
  loc <- Rfast::nth(as.matrix(df[row,1:lim]), x+1, index.return = TRUE, descending = F)
  return(colnames(df)[loc])
}

#function to return challenger whisky name - returns the largest distance from the row
get_challenger <- function(x, row, df, lim){
  loc <- Rfast::nth(as.matrix(df[row,1:lim]), x, index.return = TRUE, descending = T)
  return(colnames(df)[loc])
}
#Create tables by cluster

cluster_distance <- function (x){ 
  #creates the subgroup filtering out rows
  distances <<- subset(dist_matrix, dist_matrix[, "cluster"] == x)
  #removes fullstops from column names - needed for comparison below
  names(distances) <- gsub("\\.", " ", names(distances))
  #creates a list of columns to keep based on matching row names to column names
  cols_to_keep <- match(row.names(distances), colnames(distances))
  #removes columns that don't also appear in a row name
  distances <- distances[, cols_to_keep]
  #captures the size of the data frame 
  limit <- ncol(distances)
  #returns the 3 recommendations and the challenger option 
  for (i in 1:nrow(distances)){
    distances$recommendation1[i] <- get_recc(1, i, distances, limit) 
    distances$recommendation2[i] <- get_recc(2, i, distances, limit) 
    distances$recommendation3[i] <- get_recc(3, i, distances, limit) 
    distances$challenger[i] <- get_challenger(1, i, distances, limit) 
  }
  #removes the distance measures leaving the recommendations behind
  distances <- distances[-c(1:limit)]
  return(distances)
}

cluster_1 <- cluster_distance(1)
cluster_2 <- cluster_distance(2)
cluster_3 <- cluster_distance(3)
cluster_4 <- cluster_distance(4)
cluster_5 <- cluster_distance(5)
cluster_6 <- cluster_distance(6)

complete_recommendations <- rbind.data.frame(cluster_1, cluster_2, cluster_3, 
                                             cluster_4, cluster_5, cluster_6)
complete_recommendations <- complete_recommendations[order(row.names(complete_recommendations)),]
rownames(complete_recommendations) <- malt_names$Malt.Name

write.table(complete_recommendations, sep=",",  file = "whisky_recommendations.csv")
