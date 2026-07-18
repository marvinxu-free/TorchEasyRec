# #!/usr/bin/env python
# # -*- encoding: utf-8 -*-
# # 已对齐项目制
# # 用户侧基础特征，比如用户id、年龄、性别等，可用于召回、粗排和精排模型。
# user_basic_fnames = ['f_user_id', 'f_user_register_year', 'f_user_register_month', 'f_user_register_day', 'f_user_register_time', 'f_user_device_model', 'f_user_device_model_head','f_ext_user_user_name',
#                     'f_user_7_d_doc_id_click_score_list', 'f_user_360_d_doc_id_click_score_list', 'f_user_1_d_doc_id_click_score_list',
#                     'f_user_7_d_doc_id_conversion_score_list', 'f_user_360_d_doc_id_conversion_score_list', 'f_user_1_d_doc_id_conversion_score_list',
#                     'f_user_360_d_doc_id_favorite_score_list', 'f_user_1_d_doc_id_favorite_score_list',
#                     'f_user_360_d_goods_cate_1_click_score_list', 'f_user_360_d_goods_cate_2_click_score_list', 'f_user_360_d_goods_current_price_10_click_score_list', 'f_user_360_d_goods_ext_brand_click_score_list', 'f_user_360_d_goods_ext_spfl_click_score_list', 'f_user_360_d_goods_ext_site_click_score_list', 'f_user_360_d_goods_cate_1_conversion_score_list', 'f_user_360_d_goods_cate_2_conversion_score_list', 'f_user_360_d_goods_current_price_10_conversion_score_list', 'f_user_360_d_goods_ext_brand_conversion_score_list', 'f_user_360_d_goods_ext_spfl_conversion_score_list', 'f_user_360_d_goods_ext_site_conversion_score_list', 'f_user_360_d_goods_cate_1_favorite_score_list', 'f_user_360_d_goods_cate_2_favorite_score_list', 'f_user_360_d_goods_current_price_10_favorite_score_list', 'f_user_360_d_goods_ext_brand_favorite_score_list', 'f_user_360_d_goods_ext_spfl_favorite_score_list', 'f_user_360_d_goods_ext_site_favorite_score_list',
#                     'f_user_1_d_goods_cate_1_click_score_list', 'f_user_1_d_goods_cate_2_click_score_list', 'f_user_1_d_goods_current_price_10_click_score_list', 'f_user_1_d_goods_ext_brand_click_score_list', 'f_user_1_d_goods_ext_spfl_click_score_list', 'f_user_1_d_goods_ext_site_click_score_list', 'f_user_1_d_goods_cate_1_conversion_score_list', 'f_user_1_d_goods_cate_2_conversion_score_list', 'f_user_1_d_goods_current_price_10_conversion_score_list', 'f_user_1_d_goods_ext_brand_conversion_score_list', 'f_user_1_d_goods_ext_spfl_conversion_score_list', 'f_user_1_d_goods_ext_site_conversion_score_list', 'f_user_1_d_goods_cate_1_favorite_score_list', 'f_user_1_d_goods_cate_2_favorite_score_list', 'f_user_1_d_goods_current_price_10_favorite_score_list', 'f_user_1_d_goods_ext_brand_favorite_score_list', 'f_user_1_d_goods_ext_spfl_favorite_score_list', 'f_user_1_d_goods_ext_site_favorite_score_list',
#                     'f_user_7_d_goods_cate_1_click_score_list', 'f_user_7_d_goods_cate_2_click_score_list', 'f_user_7_d_goods_current_price_10_click_score_list', 'f_user_7_d_goods_ext_brand_click_score_list', 'f_user_7_d_goods_ext_spfl_click_score_list', 'f_user_7_d_goods_ext_site_click_score_list', 'f_user_7_d_goods_cate_1_conversion_score_list', 'f_user_7_d_goods_cate_2_conversion_score_list', 'f_user_7_d_goods_current_price_10_conversion_score_list', 'f_user_7_d_goods_ext_brand_conversion_score_list', 'f_user_7_d_goods_ext_spfl_conversion_score_list', 'f_user_7_d_goods_ext_site_conversion_score_list', 'f_user_7_d_goods_cate_1_favorite_score_list', 'f_user_7_d_goods_cate_2_favorite_score_list', 'f_user_7_d_goods_current_price_10_favorite_score_list', 'f_user_7_d_goods_ext_brand_favorite_score_list', 'f_user_7_d_goods_ext_spfl_favorite_score_list', 'f_user_7_d_goods_ext_site_favorite_score_list'
#                     ]

# # 候选侧基础特征，比如候选id、类目等，可用于召回、粗排和精排模型。
# item_basic_fnames = ['f_doc_id', 'f_goods_title', 'f_goods_title_terms', 'f_goods_cate_1', 'f_goods_cate_2', 'f_goods_tags_terms', 'f_goods_praise_cnt_10', 'f_goods_comment_cnt_10', 'f_goods_pub_time_month_day', 'f_goods_pub_time_hour', 'f_goods_pub_time_month', 'f_goods_comment_cnt', 'f_goods_comment_cnt_1', 'f_goods_current_price', 'f_goods_current_price_10', 'f_goods_current_price_1000', 'f_goods_praise_cnt', 'f_goods_praise_cnt_1', 'f_goods_ext_username', 'f_goods_ext_sub_title', 'f_goods_ext_brand', 'f_goods_ext_cai_cnt', 'f_goods_ext_spfl', 'f_goods_pub_time', 'f_goods_tags', 'f_goods_ext_site', 'f_goods_ext_username_terms', 'f_goods_ext_sub_title_terms', 'f_goods_current_price_1', 'f_goods_cai_cnt_1', 'f_goods_cai_cnt_10']

# # 父候选侧基础特征，比如父候选id、父候选类目等，可用于召回、粗排和精排模型。
# parent_item_basic_fnames = []

# # 上下文基础特征，比如时间、设备型号、网络状态等，只可用于精排模型。
# context_basic_fnames = ['f_fake_context_id']

# # Viking专用上下文特征，即context_id对应的fname，只可用于召回和粗排模型。
# viking_context_fnames = ['f_fake_context_id']

# # 组合特征，比如(用户年龄，候选类目)等，只可用于精排模型。
# combine_fnames = []

# # IPS非match类候选统计特征，比如候选在最近一天内的点击数、转化数和后验点击率等特征。
# ips_non_match_item_fnames = []

# # IPS非match类用户统计特征，比如用户在最近一天内点击过的候选id列表等。
# ips_non_match_user_fnames = []

# # IPS match类统计特征，指Rossetta抽取方法为HasMatch和TobInstanceProfileMatch等IPS特征。
# ips_match_fnames = []

# # IPS特征全集，只可直接用于精排模型
# ips_fnames = ips_non_match_item_fnames + ips_non_match_user_fnames + ips_match_fnames

user_slots = ['f_user_id',
 'f_ext_user_user_name',
 'f_user_device_model',
 'f_user_device_model_head',
 'f_user_register_time',
 'f_user_register_year',
 'f_user_register_month',
 'f_user_register_day',
 'f_user_360_d_doc_id_click_score_list',
 'f_user_1_d_doc_id_click_score_list',
 'f_user_7_d_doc_id_click_score_list',
 'f_user_360_d_doc_id_conversion_score_list',
 'f_user_1_d_doc_id_conversion_score_list',
 'f_user_7_d_doc_id_conversion_score_list',
 'f_user_360_d_doc_id_favorite_score_list',
 'f_user_1_d_doc_id_favorite_score_list',
 'f_user_360_d_goods_cate_1_click_score_list',
 'f_user_1_d_goods_cate_1_click_score_list',
 'f_user_7_d_goods_cate_1_click_score_list',
 'f_user_360_d_goods_cate_1_conversion_score_list',
 'f_user_1_d_goods_cate_1_conversion_score_list',
 'f_user_7_d_goods_cate_1_conversion_score_list',
 'f_user_360_d_goods_cate_1_favorite_score_list',
 'f_user_1_d_goods_cate_1_favorite_score_list',
 'f_user_7_d_goods_cate_1_favorite_score_list',
 'f_user_360_d_goods_cate_2_click_score_list',
 'f_user_1_d_goods_cate_2_click_score_list',
 'f_user_7_d_goods_cate_2_click_score_list',
 'f_user_360_d_goods_cate_2_conversion_score_list',
 'f_user_1_d_goods_cate_2_conversion_score_list',
 'f_user_7_d_goods_cate_2_conversion_score_list',
 'f_user_360_d_goods_cate_2_favorite_score_list',
 'f_user_1_d_goods_cate_2_favorite_score_list',
 'f_user_7_d_goods_cate_2_favorite_score_list',
 'f_user_360_d_goods_current_price_10_click_score_list',
 'f_user_1_d_goods_current_price_10_click_score_list',
 'f_user_7_d_goods_current_price_10_click_score_list',
 'f_user_360_d_goods_current_price_10_conversion_score_list',
 'f_user_1_d_goods_current_price_10_conversion_score_list',
 'f_user_7_d_goods_current_price_10_conversion_score_list',
 'f_user_360_d_goods_current_price_10_favorite_score_list',
 'f_user_1_d_goods_current_price_10_favorite_score_list',
 'f_user_7_d_goods_current_price_10_favorite_score_list',
 'f_user_360_d_goods_ext_brand_click_score_list',
 'f_user_1_d_goods_ext_brand_click_score_list',
 'f_user_7_d_goods_ext_brand_click_score_list',
 'f_user_360_d_goods_ext_brand_conversion_score_list',
 'f_user_1_d_goods_ext_brand_conversion_score_list',
 'f_user_7_d_goods_ext_brand_conversion_score_list',
 'f_user_360_d_goods_ext_brand_favorite_score_list',
 'f_user_1_d_goods_ext_brand_favorite_score_list',
 'f_user_7_d_goods_ext_brand_favorite_score_list',
 'f_user_360_d_goods_ext_spfl_click_score_list',
 'f_user_1_d_goods_ext_spfl_click_score_list',
 'f_user_7_d_goods_ext_spfl_click_score_list',
 'f_user_360_d_goods_ext_spfl_conversion_score_list',
 'f_user_1_d_goods_ext_spfl_conversion_score_list',
 'f_user_7_d_goods_ext_spfl_conversion_score_list',
 'f_user_360_d_goods_ext_spfl_favorite_score_list',
 'f_user_1_d_goods_ext_spfl_favorite_score_list',
 'f_user_7_d_goods_ext_spfl_favorite_score_list',
 'f_user_360_d_goods_ext_site_click_score_list',
 'f_user_360_d_goods_title_terms_click_score_list',
 'f_user_360_d_goods_tags_terms_click_score_list',
 'f_user_360_d_goods_ext_sub_title_terms_click_score_list',
 'f_user_360_d_goods_cate_3_click_score_list',
 'f_user_360_d_goods_doc_type_click_score_list',
 'f_user_1_d_goods_ext_site_click_score_list',
 'f_user_1_d_goods_title_terms_click_score_list',
 'f_user_1_d_goods_tags_terms_click_score_list',
 'f_user_1_d_goods_ext_sub_title_terms_click_score_list',
 'f_user_1_d_goods_cate_3_click_score_list',
 'f_user_1_d_goods_doc_type_click_score_list',
 'f_user_7_d_goods_ext_site_click_score_list',
 'f_user_7_d_goods_title_terms_click_score_list_v1',
 'f_user_7_d_goods_tags_terms_click_score_list_v1',
 'f_user_7_d_goods_ext_sub_title_terms_click_score_list_v1',
 'f_user_7_d_goods_cate_3_click_score_list',
 'f_user_7_d_goods_doc_type_click_score_list_v1',
 'f_user_360_d_goods_ext_site_conversion_score_list',
 'f_user_360_d_goods_ext_sub_title_terms_conversion_score_list',
 'f_user_360_d_goods_title_terms_conversion_score_list',
 'f_user_360_d_goods_tags_terms_conversion_score_list',
 'f_user_360_d_goods_cate_3_conversion_score_list',
 'f_user_360_d_goods_doc_type_conversion_score_list',
 'f_user_1_d_goods_ext_site_conversion_score_list',
 'f_user_1_d_goods_ext_sub_title_terms_conversion_score_list',
 'f_user_1_d_goods_title_terms_conversion_score_list',
 'f_user_1_d_goods_tags_terms_conversion_score_list',
 'f_user_1_d_goods_cate_3_conversion_score_list',
 'f_user_1_d_goods_doc_type_conversion_score_list',
 'f_user_7_d_goods_ext_site_conversion_score_list',
 'f_user_7_d_goods_ext_sub_title_terms_conversion_score_list',
 'f_user_7_d_goods_title_terms_conversion_score_list',
 'f_user_7_d_goods_tags_terms_conversion_score_list',
 'f_user_7_d_goods_cate_3_conversion_score_list',
 'f_user_7_d_goods_doc_type_conversion_score_list',
 'f_user_360_d_goods_ext_site_favorite_score_list',
 'f_user_360_d_goods_title_terms_favorite_score_list',
 'f_user_360_d_goods_tags_terms_favorite_score_list',
 'f_user_360_d_goods_ext_sub_title_terms_favorite_score_list',
 'f_user_360_d_goods_cate_3_favorite_score_list',
 'f_user_360_d_goods_doc_type_favorite_score_list',
 'f_user_1_d_goods_ext_site_favorite_score_list',
 'f_user_1_d_goods_title_terms_favorite_score_list',
 'f_user_1_d_goods_tags_terms_favorite_score_list',
 'f_user_1_d_goods_ext_sub_title_terms_favorite_score_list',
 'f_user_1_d_goods_cate_3_favorite_score_list',
 'f_user_1_d_goods_doc_type_favorite_score_list',
 'f_user_7_d_goods_ext_site_favorite_score_list',
 'f_user_7_d_goods_title_terms_favorite_score_list',
 'f_user_7_d_goods_tags_terms_favorite_score_list',
 'f_user_7_d_goods_ext_sub_title_terms_favorite_score_list',
 'f_user_7_d_goods_cate_3_favorite_score_list',
 'f_user_7_d_goods_doc_type_favorite_score_list']

group_slots = ['f_doc_id',
 'f_goods_current_price',
 'f_goods_pub_time',
 'f_goods_title',
 'f_goods_praise_cnt',
 'f_goods_comment_cnt',
 'f_goods_tags',
 'f_goods_ext_brand',
 'f_goods_ext_username',
 'f_goods_ext_cai_cnt',
 'f_goods_ext_sub_title',
 'f_goods_ext_site',
 'f_goods_ext_spfl',
 'f_goods_cate_1',
 'f_goods_cate_2',
 'f_goods_title_terms',
 'f_goods_ext_username_terms',
 'f_goods_ext_sub_title_terms',
 'f_goods_pub_time_month',
 'f_goods_pub_time_month_day',
 'f_goods_pub_time_hour',
 'f_goods_tags_terms',
 'f_goods_current_price_1',
 'f_goods_current_price_10',
 'f_goods_current_price_1000',
 'f_goods_praise_cnt_1',
 'f_goods_praise_cnt_10',
 'f_goods_cai_cnt_1',
 'f_goods_cai_cnt_10',
 'f_goods_comment_cnt_1',
 'f_goods_comment_cnt_10']