package macro;

import java.util.*;
import java.io.*;
import star.common.*;
import star.base.neo.*;
import star.meshing.*;

public class Geometry_Janitor extends StarMacro {

    public void execute() {
        Simulation sim = getActiveSimulation();
        String configFile = "geometry_swap.csv"; 
        
        File file = new File(configFile);
        if (!file.exists()) {
            sim.println("No CAD swap config found. Proceeding with existing geometry.");
            return;
        }

        try (BufferedReader br = new BufferedReader(new FileReader(file))) {
            String line;
            
            PartImportManager importManager = sim.get(PartImportManager.class);
            SimulationPartManager simPartManager = sim.get(SimulationPartManager.class);
            TagManager tagManager = sim.get(TagManager.class);
            MeshOperationManager opManager = sim.get(MeshOperationManager.class);

            while ((line = br.readLine()) != null) {
                String[] data = line.split(",");
                if (data.length < 3) continue;

                String cadPath = data[0].trim();
                String targetName = data[1].trim();
                String targetTag = data[2].trim();

                sim.println("========================================");
                sim.println(">> Processing target: " + targetName);

                // 1. DELETE EXISTING PART TO PREVENT DUPLICATES
                if (simPartManager.has(targetName)) {
                    sim.println("   -> Deleting existing part: " + targetName);
                    simPartManager.removeObjects(simPartManager.getPart(targetName));
                }

                // 2. EXTRACT RAW NAME FROM FILENAME
                File cadFile = new File(cadPath);
                String rawName = cadFile.getName();
                if (rawName.indexOf('.') > 0) {
                    rawName = rawName.substring(0, rawName.lastIndexOf('.'));
                }

                // 3. IMPORT USING EXACT V20 API PROTOCOL
                sim.println("   -> Importing: " + cadPath);
                importManager.importCadParts2(
                    new StringVector(new String[] {cadPath}), 
                    "SharpEdges", 30.0, 2, true, 1.0E-5, true, false, false, false, true, true, false
                );

                // 4. GRAB THE CADPART, RENAME, AND TAG
                try {
                    CadPart newPart = (CadPart) simPartManager.getPart(rawName);
                    newPart.setPresentationName(targetName);
                    sim.println("   -> Renamed to: " + targetName);

                    UserTag myTag;
                    if (tagManager.has(targetTag)) {
                        myTag = (UserTag) tagManager.getObject(targetTag);
                    } else {
                        sim.println("   -> Creating new Tag: " + targetTag);
                        myTag = tagManager.createNewUserTag(targetTag);
                    }

                    tagManager.setTags(newPart, new ArrayList<>(Arrays.<Tag>asList(myTag)));
                    sim.println("   -> Successfully tagged as: " + targetTag);

                    // 5. AUTO-HANDLE MIRRORED WHEELS
                    if (targetName.contains("Left Wheel")) {
                        String mirrorName = targetName + "_Mirror";
                        if (simPartManager.has(mirrorName)) {
                            MeshOperationPart mirrorPart = (MeshOperationPart) simPartManager.getPart(mirrorName);
                            String rightName = targetName.replace("Left", "Right");
                            mirrorPart.setPresentationName(rightName);
                            tagManager.setTags(mirrorPart, new ArrayList<>(Arrays.<Tag>asList(myTag)));
                            sim.println("   -> Successfully caught, renamed, and tagged mirror part: " + rightName);
                        }
                    }
                } catch (Exception e) {
                    sim.println("[ERROR] Failed to grab or tag part. Looking for: " + rawName);
                    e.printStackTrace();
                }
            }
            
            sim.println("========================================");
            sim.println("Geometry swapped. Executing all Mesh Operations...");
            opManager.executeAll();
            sim.println("Geometry Janitor complete. Ready for solver.");

        } catch (Exception e) {
            sim.println("[CRITICAL ERROR] processing geometry swap: " + e.getMessage());
            e.printStackTrace();
        }
    }
}