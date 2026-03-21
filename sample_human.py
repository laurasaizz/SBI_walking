import opensim as osm
import numpy as np
#takes as parameters the name of the output file, prints the scaled model as a osim file into the folder and returns the model internally
def sample_human(name_of_return_file: str):
    model_in = "gait2392_simbody.osim" #standard model

    # parameters of the bivariat lognormal distributed properties height and mass
    mu_h, sigma_h = 0.562, 0.039    
    mu_w, sigma_w = 4.354, 0.165   
    corr = 0.47                    

    #sampling bivariate values
    cov = corr * sigma_h * sigma_w
    ln_h, ln_w = np.random.multivariate_normal([mu_h, mu_w], [[sigma_h**2, cov], [cov, sigma_w**2]])

    #back transformation
    new_height = np.exp(ln_h)
    new_mass = np.exp(ln_w)
    force_factor = (new_mass/75.337)**(2/3) #explaination: muscle force = constant1 * area of muscle, area of muscle = constant2 * r^2, r = density factor * mass^{1/3}
    height_factor = new_height/1.7
    
    #loading tool
    scale_tool = osm.ScaleTool()

    #default settings for the scale tool to only scale based on new mass and height
    scale_tool.getGenericModelMaker().setModelFileName(model_in)
    scale_tool.getMarkerPlacer().setApply(False) 
    scaler = scale_tool.getModelScaler()
    scaler.setApply(True)
    order = osm.ArrayStr()
    order.append("manualScale")
    scaler.setScalingOrder(order)
    scale_tool.setSubjectMass(new_mass)
    scaler.setOutputModelFileName("temp_result.osim")

    #load the segments of the body (called bodies) and load them into the scaler
    base_model = osm.Model(model_in)
    body_set = base_model.getBodySet()
    scale_set = scaler.getScaleSet()
    scale_set.clearAndDestroy() 

    for i in range(body_set.getSize()):
        body_name = body_set.get(i).getName()
        new_scale = osm.Scale()
        new_scale.setName(body_name)
        new_scale.setSegmentName(body_name)
        new_scale.setScaleFactors(osm.Vec3(height_factor, height_factor, height_factor)) #uniform scaling in all directions
        new_scale.setApply(True)
        scale_set.adoptAndAppend(new_scale)

 
    scale_tool.run()
 
    #updating muscle fiber and tendon lengths because the scale tool does not overwrite those; muscle fiber length and tendon slack (length of tendon where it starts to develop force when streched further) depend roughly linearly on height, muscle force, as explained above, see definition of force_factor
    final_model = osm.Model("temp_result.osim")
    final_model.initSystem() #internal consistency of the model

    for m in range(final_model.getMuscles().getSize()):
        musc = final_model.getMuscles().get(m)

        # 1. Geometrische Skalierung (Faser- und Sehnenlänge)
        musc.setOptimalFiberLength(musc.getOptimalFiberLength() * height_factor)
        musc.setTendonSlackLength(musc.getTendonSlackLength() * height_factor)
        
        # 2. Kraft-Skalierung (Allometrisch nach Masse^2/3)
        old_force = musc.getMaxIsometricForce()
        new_force = old_force * force_factor
        musc.setMaxIsometricForce(new_force)

    final_model.printToXML(name_of_return_file)
    return final_model, new_height

def adjust_pelvis_ty_by_height(
    input_mot: str,
    sampled_height: float,
    output_mot: str,
    reference_height: float = 1.70,
    pelvis_ty_column: str = "pelvis_ty",
) -> str:

    if sampled_height <= 0:
        raise ValueError("sampled_height muss > 0 sein.")
    if reference_height <= 0:
        raise ValueError("reference_height muss > 0 sein.")

    table = osm.TimeSeriesTable(input_mot)
    labels = table.getColumnLabels()

    # Robust gegen unterschiedliche Python-Bindings:
    if isinstance(labels, (tuple, list)):
        label_list = list(labels)
    else:
        label_list = [labels.get(i) for i in range(labels.size())]

    if pelvis_ty_column not in label_list:
        raise ValueError(
            f"Spalte '{pelvis_ty_column}' nicht gefunden.\n"
            f"Vorhandene Spalten: {label_list}"
        )

    scale_factor = sampled_height / reference_height

    col = table.updDependentColumn(pelvis_ty_column)
    for i in range(col.size()):
        col[i] = col[i] * scale_factor

    # inDegrees-Metadatum sicher setzen
    try:
        table.removeTableMetaDataKey("inDegrees")
    except Exception:
        pass

    table.addTableMetaDataString("inDegrees", "yes")

    osm.STOFileAdapter.write(table, output_mot)
    return output_mot



