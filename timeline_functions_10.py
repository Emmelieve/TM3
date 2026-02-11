# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 14:05:23 2025

@author: eldenbreejen
"""
import numpy as np
import pandas as pd
import plotly.express as px
import os
from plotly.subplots import make_subplots
from event_detection_functions_15 import get_ventilation_mode

def create_timelines(dataset, tabular_data):
    '''
    Function to generate scrollable timelines of ICU admissions with RR, Vt, PF ratio, FiO2, ventilation mode and events.

    Parameters
    ----------
    dataset : str
        Name of dataset and folder with patient files.
    tabular_data : DataFrame
        Dataframe with tabular and events/controls per record.

    Returns
    -------
    None.

    '''

    # Read patient file names
    data_files                      = os.listdir(f'P:/Emmelieve/{dataset}')
    
    # Convert datatime columns
    timestamp_columns               = ['AddmissionDate', 'DemissionDate','death'] + tabular_data.filter(regex="^event").columns.tolist()
    tabular_data[timestamp_columns] = tabular_data[timestamp_columns].apply(pd.to_datetime, format="%Y-%m-%d %H:%M:%S")
    
    # Create timeline for each patient file
    for file in data_files:
         
         # Read patient file
         data            = pd.read_csv(f'P:/Emmelieve/{dataset}/{file}', index_col=0)
         data.index.name = 'Timestamp'
         data.index      = pd.to_datetime(data.index, format="%Y-%m-%d %H:%M:%S")
         
         # Get tabular and event data
         patient_id      = file[:-4]
         patient_info    = tabular_data[tabular_data.PatientID == int(patient_id)]
         
         # Remove data from before and after ICU admission
         admission_time  = patient_info.AddmissionDate.iloc[0]
         demission_time  = patient_info.DemissionDate.iloc[0]
         mask            = (data.index >= admission_time) & (data.index <= demission_time)
         data            = data.loc[mask]
         
         # Get time periods with ventilation by Hamilton ventilator
         hamilton        = data['Adem minuutvolume (contr) Hamilton C6'].notna()
         
         # Get ventilation mode segments, based on sponteaneous ventilation frequency
         ventilation_mode_df    = get_ventilation_mode(data['Ademfrequentie spontaan (contr)'], hamilton)
         
         # Get ventilation mode, based on settings
         ventilation_mode_inst  = pd.Series(index=data.index)
         
         for parameter, mode in zip(['P Control', 'P insp (contr)', 'Pressure support (inst)'], ['PCMV', 'ASV', 'Spont']):
             try:
               ventilation_mode_inst = ventilation_mode_inst.mask(data[parameter].notna(), mode) 
             except:
                 continue
             
         ventilation_mode_inst = ventilation_mode_inst.fillna('NaN')
         
         groups     = (ventilation_mode_inst != ventilation_mode_inst.shift()).cumsum()
         segments   = ventilation_mode_inst.groupby(groups).apply(lambda g: (g.index[0], 
                                                                      g.index[-1] + pd.Timedelta(minutes=1), 
                                                                      g.iloc[0], 
                                                                      None, 
                                                                      'Ventilation mode inst'))
         ventilation_mode_inst_df          = pd.DataFrame(segments.tolist(), columns=['Start', 'End', 'Label', 'Value', 'Parameter'])
         ventilation_mode_inst_df["Label"] = ventilation_mode_inst_df["Label"].replace("NaN", np.nan)
         
         
         # Get FiO2 segments, higher or lower than 40%
         fio2        = data['FiO2 (inst)']
         groups      = (fio2 != fio2.shift()).cumsum()
         segments    = fio2.groupby(groups).apply(lambda g: (g.index[0], 
                                                               g.index[-1] + pd.Timedelta(minutes=1), 
                                                               g.iloc[0], 
                                                               'FiO2'))
         fio2_df     = pd.DataFrame(segments.tolist(), columns=['Start', 'End', 'Value', 'Parameter'])
         fio2_df     = fio2_df[fio2_df.Value.notnull()]
         
         fio2_df['Label'] = fio2_df.Value.apply(lambda x: '>= 40%' if x >= 40 else '< 40%') 
         
         
         # Get PF ratio segments
         po2             = data['Art. PO2']
         pf_ratio        = po2 / fio2
         pf_ratio        = pf_ratio[pf_ratio.notnull()]
         conditions      = [pf_ratio <= 13.3/100,
                            (pf_ratio > 13.3/100) & (pf_ratio <= 26.6/100),
                            (pf_ratio > 26.6/100) & (pf_ratio <= 39.9/100),
                            (pf_ratio > 39.9/100)]
         labels          = ['<= 13.3 kPa', '13.3-26.6 kPa', '26.6-39.9 kPa', '> 39.9 kPa']
         pf_ratio_cat    = np.select(conditions, labels, default=np.nan)
         
         pf_ratio_df = {'Start': pf_ratio.index,
                        'End': pf_ratio.index + pd.Timedelta(minutes=30),
                        'Label': pf_ratio_cat,
                        'Value': pf_ratio.values*100,
                        'Parameter': ['PF ratio' for i in pf_ratio]
             }
         pf_ratio_df = pd.DataFrame(pf_ratio_df)
         
         # Events list
         events = [
             (admission_time, "Admission"),
             (demission_time, "Demission")
         ]
         
         if pd.notna(patient_info.death.iloc[0]):
             events.append((patient_info.death.iloc[0], 'Death'))
         
         events_resp = patient_info.filter(regex=r"^event_\d+")
         events_resp = events_resp.stack().tolist()
         
         if len(events_resp) > 0:
             for event in events_resp:
                 events.append((event, 'Respiratory deterioration'))
                 
         events_contr = patient_info.filter(regex=r"^event_control")
         events_contr = events_contr.stack().tolist()
        
         if len(events_contr) > 0:
             for event in events_contr:
                 events.append((event, 'Control event'))
                 
         # Plot timeline
         
         # Join dfs
         segments_df = pd.concat([ventilation_mode_df, ventilation_mode_inst_df, fio2_df, pf_ratio_df])
         
         # Mapping colors for labels
         labels = ['assisted', 'controlled', 'ASV', 'PCMV', 'Spont', '< 40%', '>= 40%', '<= 13.3 kPa', '13.3-26.6 kPa', '26.6-39.9 kPa', '> 39.9 kPa']
         colors = px.colors.qualitative.Plotly[0:7] + ['#5276BA' , '#7EB8CF' , '#ABE2DD' , '#DAF3E8' ]
         color_map = {}
         
         for i, label in zip(range(len(labels)), labels):
             color_map[label] = colors[i]
         
         # Create timeline
         fig1 = px.timeline(segments_df, 
                          x_start="Start", 
                          x_end="End", 
                          y='Parameter', 
                          color='Label',
                          hover_name='Parameter',
                          hover_data=['Value','Start', 'End'],
                          color_discrete_map= color_map)
            
         # Create subplot respiratory rate, tidal volume, and V'CO2 traces
         # Convert index to column (for compatibility with Plotly Express)
         try:
             continuous_param = ["Ademfrequentie spontaan (contr)", "Tidal Volume exp (contr)", "V'CO2"]
             data_subset = data[continuous_param].copy()
             vco2 = True
             #data_subset["V'CO2"] = data_subset["V'CO2"]/10
         except:
             continuous_param = ["Ademfrequentie spontaan (contr)", "Tidal Volume exp (contr)"]
             data_subset = data[continuous_param].copy()
             vco2 = False
             
         #data_subset = data_subset.rolling(window=10, center=True).mean()      # smooth trace
         continuous_df = data_subset.reset_index().rename(columns={"index": "Timestamp"})
    
         # Melt dataframe in order to plot multiple traces
         continuous_df = continuous_df.melt(id_vars="Timestamp", 
                                            value_vars=continuous_param,
                                            var_name="Parameter", 
                                            value_name="Value")
         
         # Plot
         fig2 = px.line(continuous_df, x="Timestamp", y="Value", color="Parameter")
    
         fig_sub = make_subplots(rows=5, 
                                 shared_xaxes=True,
                                 specs=[
                                        [{"secondary_y": True}],  # rij 1
                                        [{}],                     # rij 2
                                        [{}],                     # rij 3
                                        [{}],                     # rij 4
                                        [{}]                      # rij 5
                                    ],
                                 row_heights=[0.5, 0.5/3, 0.5/3, 0.5/6, 0.5/6],
                                 vertical_spacing=0.03
                                 )
         
         fig_sub.add_trace(fig2['data'][0], row=1, col=1, secondary_y=False)
         for trace in fig2['data'][1::]:
             fig_sub.add_trace(trace, row=1, col=1, secondary_y=True)
         
         subplot_order = reversed([5, 5, 4, 4, 4, 3, 3, 2, 2, 2, 2])
         for label, subplot in zip(reversed(labels), subplot_order):
             for trace in fig1['data']:
                 if trace.legendgroup == label:
                     fig_sub.add_trace(trace, row=subplot, col=1)
                     
        
         fig_sub.update_xaxes(type='date')
         fig_sub.update_layout(barmode='overlay')
         
         fig_sub.update_yaxes(title_text="Respiratory rate", row=1, col=1, secondary_y=False)
         if vco2 == True:
             fig_sub.update_yaxes(title_text="Tidal volume (ml) \n V'CO2 (ml/min)", row=1, col=1, secondary_y=True)
         else:
             fig_sub.update_yaxes(title_text="Tidal volume (ml)", row=1, col=1, secondary_y=True)
    
         # Update legend group
         for trace in fig_sub.data:
             if trace.legendgroup in continuous_param:
                 trace.update(legendgroup = 'Continuous parameters')
             else:
                 group = trace.y[0] if isinstance(trace.y, np.ndarray) else trace.y
                 trace.update(legendgroup = group)
    
        # Plot events and controls on timeline
         for timestamp, label in events:
           timestamp = timestamp.tz_localize("Europe/Amsterdam")
           fig_sub.add_vline(x=timestamp.timestamp() * 1000, annotation_text=label, row=1, col=1)
           for i in range(2,6):
               fig_sub.add_vline(x=timestamp.timestamp() * 1000, row=i, col=1)
           
             
         # Make it scrollable
         fig_sub.update_layout(
             xaxis5=dict(
                 range=[
                     segments_df['Start'].min(),  # begin van de scrollbare range
                     segments_df['Start'].min() + pd.Timedelta(hours=6)  # eind van zichtbare range
                 ],
                 rangeslider=dict(visible=True),  # sliders onderaan
                 type="date"
             )
         )
         
         fig_sub.write_html(f"P:/Emmelieve/timelines_{dataset}/{patient_id}.html")
    
         
        

 
    

