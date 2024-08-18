USE UNIVERSIDAD;

create table ReduccionPCA(
Id int primary key auto_increment,
Fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
Porcentaje varchar(200),
Numero_dimenciones varchar(200),
Perdida_Info varchar(200)
);
#drop table ReduccionPCA;
SELECT *FROM ReduccionPCA