# ===================================================================
# 最终美化版 V5.28 (Strain): 处理0值色块为白色
# ===================================================================

# 1. 加载所需的库
# -------------------------------------------------------------------
library(tidyverse)
library(stringr)
library(ggnewscale)

# 2. 定义分组规则和顺序
# -------------------------------------------------------------------
group_starts <- c(
  "Liver" = "attenuation_coefficient_qbox",
  "CGM"   = "cgm_mean",
  "Body"  = "bmi"
)
group_order <- names(group_starts)

# 3. 读取并整合数据 (修正后的变换逻辑)
# -------------------------------------------------------------------
pathway_data <- read.csv("pathway_Bonferroni.csv")
rename_data <- read.csv("rename_V2.csv")

group_starts_df <- enframe(group_starts, name = "group", value = "Original.Name")
rename_data_with_groups <- rename_data %>%
  left_join(group_starts_df, by = "Original.Name") %>%
  fill(group, .direction = "down") %>%
  filter(!is.na(group))

# 使用 right_join 以保留 rename_data 中的所有条目
data <- pathway_data %>%
  right_join(rename_data_with_groups, by = c("phenotype" = "Original.Name"))

# ==================== CORRECTED LOGIC START ====================
# Apply sqrt transformation to the total, then distribute by proportion
data <- data %>%
  mutate(
    # Handle potential NA values first (coalesce will turn NAs from the right_join into 0)
    pos_val = coalesce(positive, 0),
    neg_val = coalesce(negative, 0),
    total_val = pos_val + neg_val,
    
    # ==================== 代码修改 1/2 ====================
    # 当 total_val 为 0 时, 将 positive_prop 设为 NA 而不是 0
    positive_prop = if_else(total_val == 0, NA_real_, pos_val / total_val),
    
    # Calculate the transformed total bar height
    total_height_transformed = sqrt(total_val),
    
    # NEW: Calculate the value/height for plotting by distributing the
    # transformed total height according to the original proportions.
    positive_plot_val = total_height_transformed * if_else(total_val == 0, 0, positive_prop), # Use original proportion for calculation
    negative_plot_val = total_height_transformed * (1 - if_else(total_val == 0, 0, positive_prop))
  )

# Define the order of groups and variables
data$group <- factor(data$group, levels = group_order)
phenotype_order <- rename_data_with_groups$Suggested.Name
data$Suggested.Name <- factor(data$Suggested.Name, levels = phenotype_order)

# Pivot the NEW plotting values into a long format
data_long <- data %>%
  pivot_longer(
    cols = c("positive_plot_val", "negative_plot_val"),
    names_to = "observation",
    values_to = "value"
  ) %>%
  # Clean up names for the legend/fill mapping
  mutate(observation = if_else(observation == "positive_plot_val", "positive", "negative"))
# ===================== CORRECTED LOGIC END =====================

# 4. 添加空白间隙 (自定义大小)
# -------------------------------------------------------------------
gap_sizes <- c(
  "Liver" = 1,
  "CGM"   = 1,
  "Body"  = 6
)

nObsType <- nlevels(as.factor(data_long$observation))
to_add <- map_df(names(gap_sizes), function(group_name) {
  num_gaps <- gap_sizes[[group_name]]
  tibble(group = rep(group_name, num_gaps * nObsType))
})

data_long <- bind_rows(data_long, to_add)


# Re-apply factor levels to guarantee correct sort order
data_long$group <- factor(data_long$group, levels = group_order)
data_long$Suggested.Name <- factor(data_long$Suggested.Name, levels = phenotype_order)

# Arrange the data based on the correct factor levels
data_long <- data_long %>%
  arrange(group, Suggested.Name)

# Assign IDs based on the correct final order
data_long$id <- rep(seq(1, nrow(data_long) / nObsType), each = nObsType)

# 5. 计算标签、基线和网格线的位置
# -------------------------------------------------------------------
label_data <- data_long %>%
  filter(!is.na(Suggested.Name)) %>%
  group_by(id, Suggested.Name) %>%
  summarize(total_value = sum(value, na.rm = TRUE), .groups = 'drop')

heatmap_data <- data_long %>%
  distinct(id, .keep_all = TRUE)

total_positions <- max(data_long$id)
angle <- 90 - 360 * (label_data$id - 0.5) / total_positions
label_data$hjust <- ifelse(angle < -90, 1, 0)
label_data$angle <- ifelse(angle < -90, angle + 180, angle)

base_data <- data_long %>%
  filter(!is.na(Suggested.Name)) %>% 
  group_by(group) %>%
  summarize(start = min(id), end = max(id), .groups = 'drop') %>%
  rowwise() %>%
  mutate(title = mean(c(start, end)))

grid_levels_orig <- c(25, 100, 250, 450)
grid_levels_trans <- sqrt(grid_levels_orig)

# Get the height of the tallest bar
max_val_trans <- max(label_data$total_value, na.rm = TRUE)

# 让色块环的高度始终是最高柱子高度的 5%
heatmap_height <- max_val_trans * 0.05

sector_height <- max_val_trans * 1.1
y_max_limit <- max_val_trans * 1.4
y_bottom_limit <- -max_val_trans * 0.9

# 6. 为视觉元素准备数据
# -------------------------------------------------------------------
sector_colors <- c("Liver" = "#E7F5E8", "CGM" = "#FDE7E7", "Body" = "#FFFBEA")
base_data <- base_data %>% mutate(color = sector_colors[group])
grid_data <- tibble(y = grid_levels_trans, label = grid_levels_orig)

# 7. 使用 ggplot2 绘图
# -------------------------------------------------------------------
p_final <- ggplot(data_long) +
  
  # Background Layers
  geom_rect(data = base_data, aes(xmin = start - 0.5, xmax = end + 0.5, ymin = 0, ymax = sector_height, fill = group), alpha = 0.5, inherit.aes = FALSE) +
  scale_fill_manual(name = "Group Backgrounds", values = sector_colors, guide = "none") +
  geom_hline(data = grid_data, aes(yintercept = y), colour = "white", linewidth = 0.5) +
  
  # Heatmap Layer
  new_scale_fill() +
  geom_tile(data = heatmap_data, aes(x = as.factor(id), y = -heatmap_height / 2, fill = positive_prop), height = heatmap_height, inherit.aes = FALSE) +
  # ==================== 代码修改 2/2 ====================
# 将 na.value 从 "transparent" 改为 "white"，让 NA 值的色块显示为白色
scale_fill_gradient2(low = "#64B5F6", mid = "white", high = "#E57373", midpoint = 0.5, guide = "none", na.value = "white") +
  
  # Bar Layer
  new_scale_fill() +
  geom_bar(
    aes(x = as.factor(id), y = value, fill = observation),
    stat = "identity",
    alpha = 1,
    width = 0.9
  ) +
  scale_fill_manual(values = c("positive" = "#E57373", "negative" = "#64B5F6"), guide = "none") +
  
  # Label Layers
  geom_text(data = label_data, aes(x = id, y = total_value + 2, label = Suggested.Name, hjust = hjust), color = "black", fontface = "bold", alpha = 0.9, size = 3.7, angle = label_data$angle, inherit.aes = FALSE) +
  
  # MODIFIED: 分组黑线的位置也变成动态的
  geom_segment(data = base_data, aes(x = start - 0.5, y = -heatmap_height - 0.1, xend = end + 0.5, yend = -heatmap_height - 0.1), colour = "black", alpha = 0.8, linewidth = 0.6, inherit.aes = FALSE) +
  
  # Theme and Coordinates
  ylim(y_bottom_limit, y_max_limit) +
  coord_polar(clip = "off") +
  theme_void() +
  theme(legend.position = "none", plot.margin = unit(c(0, 0, 0, 0), "cm"))

# 8. 保存并显示图形
# -------------------------------------------------------------------
ggsave(p_final, file = "plot.png", width = 14, height = 14, dpi = 300, bg = "white")
print(p_final)