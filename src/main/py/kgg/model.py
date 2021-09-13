

class KGProperty(object):
    def __init__(self, _entity_id, _property, _value):
        self.entity_id = _entity_id
        self.property = _property
        self.value = _value


class KGRelation(object):
    def __init__(self, _relation_id, _relation_name):
        self.relation_id = _relation_id
        self.relation_name = _relation_name


class KGEntity(object):
    def __init__(self, _entity_id, _entity_name, _property_list):
        self.entity_id = _entity_id
        self.entity_name = _entity_name
        self.property_list = _property_list


class KGEntityRelation(object):
    def __init__(self, _src_entity_id, _relation_id, _obj_entity_id):
        self.src_entity_id = _src_entity_id
        self.relation_id = _relation_id
        self.obj_entity_id = _obj_entity_id
