import opensim as osim

model = osim.Model("gait2392_simbody.osim")
state = model.initSystem()

table = osim.TimeSeriesTable("normal.mot")

# convert degrees if necessary
if table.hasTableMetaDataKey("inDegrees"):
    if table.getTableMetaDataString("inDegrees") == "yes":
        osim.TableUtilities.convertDegreesToRadians(table)

states = osim.StatesTrajectory.createFromStatesTable(model, table)

for state in states:
    model.realizePosition(state)
    com = model.calcMassCenterPosition(state)
    print(state.getTime(), com)