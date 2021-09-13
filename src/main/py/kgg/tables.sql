-- 一、业务表
CREATE TABLE `tb_ent_stock` (
  `id` int(10) NOT NULL,
  `stock_code` varchar(16) NOT NULL,
  `stock_name` varchar(16) NOT NULL,
)
ENGINE=InnoDB DEFAULT CHARSET=utf8;


-- 二、中间表
-- 初审三元组表
DROP TABLE IF EXISTS `tb_kg_ie`;
CREATE TABLE `tb_rdf_value` (
  `id` int(10) NOT NULL,
  `entity` varchar(16) NOT NULL,
  `key` varchar(16) NOT NULL,
  `value` varchar(10) NOT NULL,
  `type` int(1) NOT NULL,
  `desc` varchar(128) DEFAULT NULL
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;



-- 三、基础信息表
-- 实体类型
CREATE TABLE `tb_ontology_type` (
  `id` int(10) NOT NULL,
  `type_id` int(10) NOT NULL,
  `value` varchar(16) DEFAULT NULL,
  `desc` varchar(128) DEFAULT NULL
  PRIMARY KEY (`id`)
)
ENGINE=InnoDB DEFAULT CHARSET=utf8;

DROP TABLE IF EXISTS `tb_entity_info`;
CREATE TABLE `tb_entity_info` (
  `entity_id` int(10) NOT NULL,
  `entity_key` varchar(16) DEFAULT NULL,
  `entity_name` varchar(16) DEFAULT NULL,
  `entity_type_id` int(10) DEFAULT NULL,
  PRIMARY KEY (`entity_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

DROP TABLE IF EXISTS `tb_property_info`;
CREATE TABLE `tb_relation_info` (
  `id` int(10) NOT NULL,
  `property_key` varchar(16) DEFAULT NULL,
  `property_name` varchar(16) DEFAULT NULL,
  PRIMARY KEY (`property_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

DROP TABLE IF EXISTS `tb_relation_info`;
CREATE TABLE `tb_relation_info` (
  `id` int(10) NOT NULL,
  `relation_key` varchar(16) DEFAULT NULL,
  `relation_name` varchar(16) DEFAULT NULL,
  PRIMARY KEY (`relation_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- 字典表
--  实体名 - 类型：中国银行，恒泰证券，xxx
DROP TABLE IF EXISTS `tb_kg_dict`;
CREATE TABLE `tb_kg_field` (
  `id` int(10) NOT NULL,
  `value` varchar(16) DEFAULT NULL,
  `tag` int(8) NOT NULL,
  `desc` varchar(128) DEFAULT NULL
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;



-- 四、标准化三元组表
-- 1.实体属性表
--  实体id
--  实体属性 id
--  属性值
DROP TABLE IF EXISTS `tb_entity_prop`;
CREATE TABLE `tb_entity_prop` (
  `id` int(10) NOT NULL,
  `entity_id` int(10) NOT NULL,
  `property_key` int(10) NOT NULL,
  `value` varchar(16) DEFAULT NULL,
  `desc` varchar(128) DEFAULT NULL
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- 2.实体关系表
--  实体 id
--  关系 id
--  实体 id
DROP TABLE IF EXISTS `tb_entity_relation`;
CREATE TABLE `tb_entity_relation` (
  `id` int(10) NOT NULL,
  `src_entity_id` int(10) NOT NULL,
  `relation_key` int(10) NOT NULL,
  `obj_entity_id` int(10) NOT NULL,
  `desc` varchar(128) DEFAULT NULL
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
