import opensim as osm


# Dateipfade
model_in = "Rajagopal2016.osim" # Name deiner Modelldatei
xml_setup = "scaling.xml" # Die XML, die du oben gepostet hast
model_out = "Rajagopal2016_scaled.osim"

# Sampling Werte, erstmal provisorisch
new_mass = 85.0
s_factor = 1.10 # 10% größer
force_factor = (new_mass/75.337)**(2/3) #Begründung: Muskelkraft = constante1 * querschnitt, querschnitt = constante2 * r^2, r = dichtefaktor * masse^{1/3}

# 1. Tool laden
scale_tool = osm.ScaleTool(xml_setup)

# 2. Grundeinstellungen überschreiben
scale_tool.getGenericModelMaker().setModelFileName(model_in)
scale_tool.getMarkerPlacer().setApply(False) # keine marker

scaler = scale_tool.getModelScaler()
scaler.setApply(True)
scaler.setScalingOrder(osm.ArrayStr("manualScale", 1)) # Auf manuell stellen
scale_tool.setSubjectMass(new_mass)
scaler.setOutputModelFileName("temp_result.osim")

# 3. Segmente dynamisch zum ScaleSet hinzufügen
# Wir laden das Modell kurz, um alle Segmentnamen zu bekommen
base_model = osm.Model(model_in)
body_set = base_model.getBodySet()
scale_set = scaler.getScaleSet()
scale_set.clearAndDestroy() # Altes Set leeren

for i in range(body_set.getSize()):
    body_name = body_set.get(i).getName()
    # Neues Scale-Objekt für diesen Body erstellen
    new_scale = osm.Scale()
    new_scale.setName(body_name)
    new_scale.setScaleFactors(osm.Vec3(s_factor))
    new_scale.setApply(True)
    scale_set.adoptAndAppend(new_scale)

# 4. Ausführen
scale_tool.run()

# 5. Muskel-Update (Fiber & Tendon Length)
# Da das ScaleTool die Muskeln nicht anpasst, machen wir das manuell:
final_model = osm.Model("temp_result.osim")
for m in range(final_model.getMuscles().getSize()):
    musc = final_model.getMuscles().get(m)

    # 1. Geometrische Skalierung (Faser- und Sehnenlänge)
    musc.setOptimalFiberLength(musc.getOptimalFiberLength() * s_factor)
    musc.setTendonSlackLength(musc.getTendonSlackLength() * s_factor)
    
    # 2. Kraft-Skalierung (Allometrisch nach Masse^2/3)
    old_force = musc.getMaxIsometricForce()
    new_force = old_force
    musc.setMaxIsometricForce(new_force)


final_model.printToXML(model_out)
print(f"Fertig! Modell gespeichert als {model_out}")



