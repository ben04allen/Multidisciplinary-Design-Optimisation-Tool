package macro;

import java.util.*;
import java.io.*;
import star.common.*;
import star.base.report.*;
import star.base.neo.*;

public class probe_parameters extends StarMacro {

    public void execute() {
        Simulation sim = getActiveSimulation();
        
        // 1. Extract Global Parameters (FILTERED BY TAG)
        List<String> parameters = new ArrayList<>();
        for (ClientServerObject obj : sim.get(GlobalParameterManager.class).getObjects()) {
            boolean isTagged = false;
            if (obj instanceof Taggable) {
                TagGroup tagGroup = ((Taggable) obj).getTagGroup();
                if (tagGroup != null) {
                    for (Tag tag : tagGroup.getObjects()) {
                        String tagName = tag.getPresentationName().toLowerCase();
                        if (tagName.contains("variable") || tagName.contains("sweep")) {
                            isTagged = true;
                            break;
                        }
                    }
                }
            }
            if (isTagged) {
                parameters.add(obj.getPresentationName());
            }
        }
        
        // 2. Extract Reports (FILTERED BY TAG)
        List<String> reports = new ArrayList<>();
        for (Report obj : sim.getReportManager().getObjects()) {
            boolean isTagged = false;
            if (obj instanceof Taggable) {
                TagGroup tagGroup = ((Taggable) obj).getTagGroup();
                if (tagGroup != null) {
                    for (Tag tag : tagGroup.getObjects()) {
                        String tagName = tag.getPresentationName().toLowerCase();
                        // Checks for "reports" or "report"
                        if (tagName.contains("reports")) {
                            isTagged = true;
                            break;
                        }
                    }
                }
            }
            if (isTagged) {
                reports.add(obj.getPresentationName());
            }
        }
        
        // 3. Extract Field Functions
        List<String> fieldFunctions = new ArrayList<>();
        for (ClientServerObject obj : sim.getFieldFunctionManager().getObjects()) {
            fieldFunctions.add(obj.getPresentationName());
        }
        
        // 4. Construct JSON string
        StringBuilder json = new StringBuilder();
        json.append("{\n");
        
        // Parameters
        json.append("  \"parameters\": [");
        for (int i = 0; i < parameters.size(); i++) {
            json.append("\"").append(parameters.get(i)).append("\"");
            if (i < parameters.size() - 1) json.append(", ");
        }
        json.append("],\n");
        
        // Reports
        json.append("  \"reports\": [");
        for (int i = 0; i < reports.size(); i++) {
            json.append("\"").append(reports.get(i)).append("\"");
            if (i < reports.size() - 1) json.append(", ");
        }
        json.append("],\n");
        
        // Field Functions
        json.append("  \"field_functions\": [");
        for (int i = 0; i < fieldFunctions.size(); i++) {
            json.append("\"").append(fieldFunctions.get(i)).append("\"");
            if (i < fieldFunctions.size() - 1) json.append(", ");
        }
        json.append("]\n");
        
        json.append("}");
        
        try {
            File file = new File("sim_metadata.json");
            FileWriter writer = new FileWriter(file);
            writer.write(json.toString());
            writer.close();
            sim.println("Successfully exported sim_metadata.json with filtered reports.");
        } catch (IOException e) {
            sim.println("Error writing JSON file.");
        }
    }
}