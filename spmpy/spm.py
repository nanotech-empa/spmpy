import os
from . import nanonispy as nap
import numpy as np
import spiepy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


class Spm:

    #Dictionary Channels
    ChannelName = ['LIR 1 omega (A)', 'LI_Demod_1_X','LI_Demod_1_Y','Z','Current','Bias','Frequency_Shift','Amplitude','Excitation','Temperature_1',
                   'Bias (V)','Bias calc (V)', 'Bias [bwd] (V)', 'Current (A)','Current [bwd] (A)','Amplitude (m)',
                   'Amplitude [bwd] (m)', 'Excitation (V)', 'Excitation [bwd] (V)', 'Frequency Shift (Hz)', 'Frequency Shift [bwd] (Hz)',
                   'LI Demod 1 X (A)','LI Demod 1 X (A) [bwd] (A)','PMT (V)','Counter 1 (Hz)','Counter_1', 'Z rel (m)', 'Z (m)','Time (s)',
                   'LI Demod 0 X (V)','LI Demod 0 Y (V)','LI Demod 3 X (A)','LI Demod 3 Y (A)','LI_Demod_3_X','LI_Demod_3_Y',
                   'Delay Sampling (s)','Delay THz1 (s)','Delay THz2 (s)','Position Phase1 (m)','Rotation1 (deg)','Rotation2 (deg)','Rotation (deg)','Index','Wavelength','Intensity']
    ChannelNickname = ['dIdV', 'dIdV','dIdV_Y','z','I','V','df','A','exc','T1',
                    'V', 'V', 'V_bw' ,'I','I_bw','A',
                    'A_bw', 'exc','exc_bw','df','df_bw',
                    'dIdV','dIdV_bw','PMT','counter','counter', 'zrel','zspec','t',
                    'EOS','EOS_Y','I_THz','I_THz_Y','I_THz','I_THz_Y',
                    'Delay','Delay1','Delay2','Phase','Rot1','Rot2','Rot','Index','Wavelength','Intensity']
    ChanneliScaling = [10**12,10**12,10**12,10**9,10**12,1,1,10**9,1,1,
                    1,1,1,10**12,10**12,10**9,
                    10**9,1,1,1,1,1,
                    1,1,1,1,10**12,10**9,1,
                    1,1,10**12,10**12,10**12,10**12,
                    10**12,10**12,10**12,10**3,1,1,1,1,1,1]
    ChannelUnit = ['pS','pS','pS','nm','pA','V','Hz','nm','V','K',
                    'V','V','V','pA','pA','nm',
                    'nm','V','V','Hz','Hz','a.u.',
                    'a.u.','V','Hz','Hz','pm','nm','s',
                    'V','V','pA','pA','pA','pA',
                    'ps','ps','ps','mm','deg','deg','deg','N','nm','']

    global SignalsListReference
    SignalsListReference = []

    for (chName,chNickname,chScaling,chUnit) in zip(ChannelName,ChannelNickname,ChanneliScaling,ChannelUnit):
        SignalsListReference.append({'ChannelName': chName, 'ChannelNickname': chNickname , 'ChannelScaling': chScaling, 'ChannelUnit': chUnit})

    del ChannelName,ChanneliScaling,ChannelUnit    
    del chName,chNickname,chScaling,chUnit     

    #Dictionary Parameters
    ParamName = ['X (m)','Y (m)','Z (m)','bias','z-controller>setpoint','scan_angle','Comments','Comment01',
                'Lock-in>Amplitude','Lock-in>Reference phase (deg)','Lock-in>Frequency (Hz)','Temperature 2>Temperature 2 (K)',
                'Current>Current (A)','Bias>Bias (V)','z-controller>tiplift (m)',
                'Parameter>Delay Sampling (s)','Parameter>Delay PP1 (s)','Parameter>Delay PP2 (s)','Parameter>Angle Rot1 (deg)','Parameter>Angle Rot2 (deg)','Parameter>Position Phase1 (m)','Parameter>Position_Phase1 (m)', 'Ext. VI 1>Laser>PP Frequency (MHz)']
    ParamNickname = ['x','y','z','V','setpoint','angle','comment','comment_spec',
                'lockin_amplitude','lockin_phase','lockin_frequency','temperature',
                'setpoint_spec','V_spec','z_offset',
                'Sampling','PP1','PP2','Rot1','Rot2','Phase','Phase_depricated','frep']
    ParamScaling = [10**9,10**9,10**9,1,10**12,1,'na','na',
                10**3,1,1,1,
                10**12,1,10**9,
                10**12,10**12,10**12,1,1,10**3,10**3,1] # na if not a numeric value
    ParamUnit = ['nm','nm','nm','V','pA','°','','',
                'mV','°','Hz','K',
                'pA','V','nm',
                'ps','ps','ps','deg','deg','mm','mm','MHz']


    global ParamListReference
    ParamListReference = []
    
    for (paName,paNickname,paScaling,paUnit) in zip(ParamName,ParamNickname,ParamScaling,ParamUnit):
        ParamListReference.append({'ParamName': paName, 'ParamNickname': paNickname, 'ParamScaling': paScaling, 'ParamUnit': paUnit})

    del ParamName,ParamScaling,ParamUnit    
    del paName,paNickname,paScaling,paUnit

    # constructor
    def __init__(self, path: str):
        """
        Spm constructor

        Parameters
        ----------
        path : str
            SPM filepath.

        Returns
        -------
        None.

        """
              
      
        # self.path = path.replace('//', '/')
        abspath = os.path.abspath(path)
        self.path = '/'.join(abspath.split('\\')[-4:])
        self.name = self.path.split('/')[-1]
        file_extension = os.path.splitext(path)[1]
        
        if file_extension == '.sxm':
            self.napImport = nap.read.Scan(path)
            self.type = 'scan'
        elif file_extension == '.dat':
            self.napImport = nap.read.Spec(path)
            self.type = 'spec'
        else:
            print('Datatype not supported.')
            return;
            
        ch = []    
        for key in self.napImport.signals:
            try:
                ch.append([d['ChannelName'] for d in SignalsListReference].index(key))
            except:
                pass;    
 
        self.SignalsList = [SignalsListReference[i] for i in ch] #List of all recorded channels
        self.channels = [c['ChannelNickname'] for c in self.SignalsList] 
        self.header = self.napImport.header
        
        
    def __repr__(self):
        return self.path
        
    #get channel
    def get_channel(self, 
                    channel: str, 
                    direction: str = 'forward', 
                    flatten: bool = False, 
                    offset: bool = False, 
                    zero: bool = False):
        """
        Returns the measurement values and the channel unit.

        Parameters
        ----------
        channel : str
            Name of the channel stored in the reference config file or in the NANONIS file.
        direction : str, optional
            Measurement direction. It can only be forward or backward. The default is 'forward'.
        flatten : bool, optional
            Substract background plane from image data. The default is False.
        offset : bool, optional
            Substract constant offset = mean value. The default is False.
        zero : bool, optional
            Substract constant offset = minimum value. The default is False.

        Returns
        -------
        tuple
            Measurement values and channel unit

        """

        
        if self.type == 'scan':
            chNum = [d['ChannelNickname'] for d in self.SignalsList].index(channel)
            im = self.napImport.signals[self.SignalsList[chNum]['ChannelName']][direction]
            im = im *self.SignalsList[chNum]['ChannelScaling']
            
            if flatten:
                if ~np.isnan(np.sum(im)):
                    im, _ = spiepy.flatten_xy(im)
                else:
                    m,n = np.shape(im)
                    i = np.argwhere(np.isnan(im))[0,0]
                    im_cut = im[:i-1,:]
                    im, _ = spiepy.flatten_xy(im_cut)
                    empty = np.full((m-i,n),np.nan)
                    im = np.vstack((im,empty))
                    
                    warnings.warn('NaN values in image')
                    #im-np.nanmean(im)
               
            if offset:
                im = im-np.mean(im)
            if zero:
                im = im+abs(np.min(im))

            unit = self.SignalsList[chNum]['ChannelUnit']
                
            return (im,unit)
        
        elif self.type == 'spec':
        
            if direction == 'backward':
                channel = channel + '_bw';
                #print(channel)
            
            chNum = [d['ChannelNickname'] for d in self.SignalsList].index(channel)
            data =  self.napImport.signals[self.SignalsList[chNum]['ChannelName']]
            data = data*self.SignalsList[chNum]['ChannelScaling']
            unit = self.SignalsList[chNum]['ChannelUnit']
            
            return (data,unit)
            
        return
    
    #get parameter            
    def get_param(self, param: str):
        """
        Returns the parameter value and the parameter unit.

        Parameters
        ----------
        param : str
            Name of the parameter stored in the reference config file or in the NANONIS file.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
 
        if any(d['ParamNickname'] == param for d in ParamListReference):
            paNum = [d['ParamNickname'] for d in ParamListReference].index(param)

            if ParamListReference[paNum]['ParamScaling'] == 'na':
                return self.napImport.header[ParamListReference[paNum]['ParamName']]
            else:
                return (float(self.napImport.header[ParamListReference[paNum]['ParamName']])*ParamListReference[paNum]['ParamScaling'],ParamListReference[paNum]['ParamUnit'])
            
        
        elif param == 'width' or param == 'height':
            # height, width = self.get_param('scan_range')
            # height, width = [height*10**9, width*10**9]
            scanfield = self.get_param('scan>scanfield')
            width, height = float(scanfield.split(';')[2])*10**9, float(scanfield.split(';')[3])*10**9
             
            return (eval(param),'nm')
        
            
        else:
            if param in self.napImport.header.keys():
                return self.napImport.header[param]
            else:
                return        
        
              
        
    # print essential parameters for plotting  
    def print_params(self, show: bool = True):
        """
        Print essential parameters for plotting

        Parameters
        ----------
        show : bool, optional
            Whether label if printed. The default is True.

        Returns
        -------
        label : list
            Set of important parameters.

        """
        
        label = []
        
        if self.type == 'scan':
            fb_enable = self.get_param('z-controller>controller status')
            fb_ctrl = self.get_param('z-controller>controller name')
            bias = self.get_param('V')
            set_point = self.get_param('setpoint')
            height = self.get_param('height')
            width = self.get_param('width')
            angle = self.get_param('angle')
            z_offset = self.get_param('z_offset')
            comment = self.get_param('comments')
            
            
                         
            if fb_enable == 'OFF':
                label.append('constant height')
                label.append('z-offset: %.3f%s' % z_offset)
                
            if np.abs(bias[0])<0.1:
                bias = list(bias)
                bias[0] = bias[0]*1000
                bias[1] = 'mV'
                bias = tuple(bias)
                
            label.append('I = %.0f%s' % set_point)    
            label.append('bias = %.2f%s' % bias)
            label.append('size: %.1f%s x %.1f%s (%.0f%s)' % (width+height+angle))
            label.append('comment: %s' % comment)
            
            
        elif self.type == 'spec':
            
            fb_enable = self.get_param('Z-Ctrl hold')
            set_point = self.get_param('setpoint_spec')
            bias = self.get_param('V_spec')
            #lockin_status = self.get_param('Lock-in>Lock-in status')
            lockin_amplitude = self.get_param('lockin_amplitude')
            lockin_phase= self.get_param('lockin_phase')
            lockin_frequency= self.get_param('lockin_frequency')
            comment = self.get_param('comment_spec')
            
                               
            #if lockin_status == 'ON':
            label.append('lockin: A = %.0f%s (θ = %.0f%s, f = %.0f%s)' % (lockin_amplitude+lockin_phase+lockin_frequency))
                 
            
            if fb_enable == 'FALSE':
                label.append('feedback on')
                
            elif fb_enable == 'TRUE':
                label.append('feedback off')
           
 
            label.append('setpoint: I = %.0f%s, V = %.1f%s' % (set_point+bias))    
            
            label.append('comment: %s' % comment)
    
        label.append('path: %s' % self.path)  
        label = '\n'.join(label)
        
        if show:
            print(label)
        
        return label
         
      
    
    # plot
    def plot(self, **params):
        """
        Plots a channel of the Spm objects

        Parameters
        ----------
        **params : TYPE
            Optional parameters for .sxm files:
                channel: Name of the channel that is plotted
                direction: Measurement direction (forward or backward)
                flatten: Boolean whether the plane subtraction
                offset: Boolean whether the constant offset (mean value) should be subtracted
                cmap: Name of color map
                clim: Lower bound and upper bound of the color limits
                log: Boolean whether the color scale is logarithmic
                show_params: Whether in the title essential parameters are shown
                show: Whether the plot is shown
                save: Whether plot is saved
                save_name: Name of the saved plot
                close_fig: Whether figure is closed at the end
                save_format: Save format
                zero: Boolean whether the constant offset (minimum value) should be subtracted

            Optional parameters for .dat files:
                channelx: Name of the X channel
                channely: Name of the Y channel
                direction: Measurement direction (forward or backward)
                log: Boolean whether the color scale is logarithmic
                loglog: Log-log plot
                show_params: Whether in the title essential parameters are shown
                show: Whether the plot is shown
                save: Whether plot is saved
                save_name: Name of the saved plot
                close_fig: Whether figure is closed at the end
                save_format: Save format

        Returns
        -------
        fig : matplotlib.pyplot.Figure
            Figure of a channel of the Spm object

        """

        #cmaps = sorted(m for m in plt.cm.datad)
        # ['Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'jet', 'jet_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spectral', 'spectral_r', 'spring', 'spring_r', 'summer', 'summer_r', 'terrain', 'terrain_r', 'winter', 'winter_r']
        
        #ml.interactive(0)

        if self.type == 'scan':
            
            if 'channel' in params:
                channel = params['channel']
            else:
                channel = self.channels[0]
                
            if 'direction' in params:
                direction = params['direction']
            else:
                direction = 'forward'
                
            if 'flatten' in params:
                flatten = params['flatten']
            else:
                flatten = False
                
            if 'offset' in params:
                offset = params['offset']
            else:
                offset = False
                
            if 'cmap' in params:
                cmap = params['cmap']
            else:
                cmap = 'gray'
                
            if 'clim' in params:
                clim = params['clim']
            else:
                clim = False
                
            if 'log' in params:
                log = params['log']
            else:
                log = False    
                
            if 'show_params' in params:
                show_params = params['show_params']
            else:
                show_params = False
                
            if 'show' in params:
                show = params['show']
            else:
                show = True
            
            if 'save' in params:
                save = params['save']
            else:
                save = False
            
            if 'save_name' in params:
                save_name = params['save_name']
            else:  
                save_name = False
                
            if 'close_fig' in params:
                close_fig = params['close_fig']
            else:
                close_fig = True

            if 'save_format' in params:
                save_format = params['save_format']
            else:
                save_format = 'png'
            
            if 'zero' in params:
                zero = params['zero']
            else:
                zero = False
                

                
            
            (chData,chUnit) = self.get_channel(channel, direction = direction, flatten=flatten, offset=offset,zero=zero);
            
            if 'vmin' in params:
                vmin = params['vmin']
            else:
                vmin = np.min(chData)

            if 'vmax' in params:
                vmax = params['vmax']
            else:
                vmax = np.max(chData)

            if direction == 'backward':
                chData = np.fliplr(chData)
            
            width = self.get_param('width')
            height = self.get_param('height')
            pix_y,pix_x = np.shape(chData)
            
            fig = plt.figure(figsize=(7,8))
            
            
            ImgOrigin = 'lower'
            if self.get_param('scan_dir') == 'down':
                ImgOrigin = 'upper'
            
            

            if log:
                im = plt.imshow(np.abs(chData), aspect = 'equal', extent = [0,width[0],0,height[0]], cmap = cmap, norm=LogNorm(), origin = ImgOrigin)
            else:
                im = plt.imshow(chData, aspect = 'equal',extent = [0,width[0],0,height[0]], vmin=vmin, vmax=vmax, cmap = cmap, origin = ImgOrigin)
            
            
            if clim:
                plt.clim(clim)
                
            
            im.axes.set_xticks([0,width[0]])
            im.axes.set_xticklabels([0,np.round(width[0],2)])
            im.axes.set_yticks([0,height[0]])
            im.axes.set_yticklabels([0,np.round(height[0],2)])
            
            if show_params:
                title = self.print_params(show = False);
            else:
                title = self.path  
                
            plt.title(title + '\n', loc='left')
            plt.xlabel('x (%s)' % width[1])
            plt.ylabel('y (%s)' % height[1])
            
            cbar = plt.colorbar(im,fraction=0.046, pad=0.02, format='%.2g',shrink = 0.5,aspect=10)
            cbar.set_label('%s (%s)' % (channel,chUnit))
            
            if show:
                plt.show()
            # else:
            #     plt.close(fig)
            
            if close_fig:
                plt.close(fig)
            
            if save:
                if save_name:
                    fname = save_name
                else:
                    fname = self.path.split('/')[-1]
                    fname = fname.split('.')[-2]
                fig.savefig(fname+'.'+save_format,dpi=500)
            
            return fig 
            

           
        elif self.type == 'spec':
            
            if 'channelx' in params:
                channelx = params['channelx']
            else:
                channelx = self.channels[0]
            
            if 'channely' in params:
                channely = params['channely']
            else:
                channely = self.channels[1]
                
            if 'direction' in params:
                direction = params['direction']
            else:
                direction = direction = 'forward'
                
            if 'log' in params:
                log = params['log']
            else:
                log = False
                
            if 'loglog' in params:
                loglog = params['loglog']
            else:
                loglog = False
                
                
            if 'show_params' in params:
                show_params = params['show_params']
            else:
                show_params = False
                
            if 'show' in params:
                show = params['show']
            else:
                show = True
                
            if 'save' in params:
                save = params['save']
            else:
                save = False

            if 'save_name' in params:
                save_name = params['save_name']
            else:
                save_name = False
                
            if 'close_fig' in params:
                close_fig = params['close_fig']
            else:
                close_fig = True
            
            if 'save_format' in params:
                save_format = params['save_format']
            else:
                save_format = 'png'
            
                
            (x_data,x_unit) = self.get_channel(channelx,direction=direction)
            (y_data,y_unit) = self.get_channel(channely,direction=direction)
            
            
            fig = plt.figure(figsize=(6,4))
            
            if log:
                plt.semilogy(x_data,np.abs(y_data))
                
            elif loglog:
                plt.loglog(np.abs(x_data),np.abs(y_data))
                
            else:
                plt.plot(x_data,y_data)          
                
           
            if show_params:
                title = self.print_params(show = False);
            else:
                title = self.path  
                
            plt.title(title + '\n', loc='left') 
                
            plt.xlabel('%s (%s)' % (channelx,x_unit))
            plt.ylabel('%s (%s)' % (channely,y_unit))
            

            if show:
                plt.show()
            # else:
            #     plt.close(fig)
            
            if close_fig:
                plt.close(fig)
            
            if save:
                if save_name:
                    fname = save_name
                else:
                    fname = self.path.split('/')[-1]
                    fname = fname.split('.')[-2]
                fig.savefig(fname+'.'+save_format,dpi=500)
            
            return fig
                             
 