from sqlalchemy import JSON, Column, ForeignKey, Integer, Unicode
from sqlalchemy.orm import as_declarative, declared_attr


@as_declarative()
class AbstractModel:
    @classmethod
    @declared_attr
    def __tablename__(cls):
        return cls.__name__.lower()


class DataFormat(AbstractModel):
    id = Column(Integer, autoincrement=True)
    name = Column(Unicode)


class ExportFormat(AbstractModel):
    id = Column(Integer, autoincrement=True)
    name = Column(Unicode)


class Model(AbstractModel):
    id = Column(Integer, autoincrement=True)


class User(AbstractModel):
    id = Column(Integer)
    name = Column(Unicode)
    apikey = Column(Unicode)


class ModelTags(AbstractModel):
    pass


class Tags(AbstractModel):
    pass


class Vectors(AbstractModel):
    pass


class Speackers(AbstractModel):
    pass


class Transkribtion(AbstractModel):
    pass


class Audio(AbstractModel):
    pass
